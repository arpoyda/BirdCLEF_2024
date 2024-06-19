import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)
import random

from sklearn.model_selection import KFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import MetricCollection, MeanMetric
from torchmetrics.classification import MultilabelAUROC

from pytorch_lightning import LightningModule, LightningDataModule, Trainer, callbacks
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar

import timm
import albumentations as A
import wandb

from compute_mels import mel_preproc


class CFG:
    model_name = 'efficientnet_b0'  # {'efficientnet_b0', 'regnety_008.pycls_in1k'}
    learning_rate: float = 1e-3     # {1e-3, 3e-3}
    epochs_num = 12                 # {7, 9, 12}
    segmentation_sec = 60           # {20, 30, 60}
    split_type = 'fold0'            # {'fold0', '80low'}
    batch_size = 96


class BirdDataset(Dataset):
    def __init__(self, 
                 df, 
                 transform=None, 
                 segm_sec=None,
                 **kwargs):        
        df = df.reset_index(drop=True)
        df.loc[:, 'npy_idx'] = df.index.tolist()
        if segm_sec is None:
            self.objidx2npyidx = df.groupby(['mel_path'], as_index=False)['npy_idx'].aggregate(lambda x: x.tolist())
        else:
            df['offset_segm'] = df['offset_seconds'].astype(int) // segm_sec
            self.objidx2npyidx = df.groupby(['mel_path', 'offset_segm'], as_index=False)['npy_idx'].aggregate(lambda x: x.tolist())
        self.npy_data = np.ascontiguousarray(np.array(df['npy_data'].values.tolist()))
        self.transform = transform
        self.pps = 32_000 // 500
        self.duration = 10

    def __len__(self):
        return len(self.objidx2npyidx)

    def __getitem__(self, idx):
        obj = self.objidx2npyidx.iloc[idx]
        rand_idx = np.random.choice(obj.npy_idx)
        offset = self.npy_data[rand_idx][0]
        x = self._read_spec(path=obj.mel_path, offset=offset)
        y = self.npy_data[rand_idx:rand_idx+2, 1:].mean(axis=0)
        return x, y
    
    def _cyclic_fill(self, x):
        if (x.shape[1] / self.pps) < self.duration:
            x = np.hstack([x for _ in range(int(self.duration / (x.shape[1] / self.pps)) + 1)])
            x = x[:, :self.pps*self.duration]
        return x
    
    def _read_spec(self, path, offset):
        x = np.load(path)
        start = int(offset * self.pps)
        end = start + self.duration * self.pps
        x = x[:, start:end]
        x = self._cyclic_fill(x)
        x = mel_preproc(x)
        x = x[np.newaxis, ...]
        if self.transform:
            x = x.transpose((1, 2, 0))
            x = self.transform(image=x)['image']
            x = x.transpose((2, 0, 1))
        return x
    

class BirdDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
            
    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=4,
                          pin_memory=False,
                          shuffle=True,
                        )

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=4,
                          pin_memory=False,
                          shuffle=False,
                        )


class CutMix:
    def __init__(self, 
                 mode: str = 'horizontal', 
                 p: float = 1.0, 
                 cuts_num: int = 1):
        assert mode in ['horizontal']
        self.mode = mode
        self.cuts_num = cuts_num
        self.p = p
  
    def apply_horizontal(self, imgs, labels):
        w = imgs.shape[-1]
        b = imgs.shape[0]
        
        alphas = np.sort(np.random.rand(self.cuts_num))
        rand_index = [np.random.permutation(b) for _ in range(self.cuts_num)]
        imgs_tomix = [imgs[idxes] for idxes in rand_index]
        labels_tomix = [labels[idxes] for idxes in rand_index]
        
        for alpha, img_tomix in zip(alphas, imgs_tomix):
            imgs[..., int(alpha*w):] = img_tomix[..., int(alpha*w):]
        
        labels = labels*alphas[0]
        for i in range(1, self.cuts_num):
            labels += labels_tomix[i-1]*(alphas[i] - alphas[i-1])
        labels +=  labels_tomix[-1] * (1 - alphas[-1])
        
        return imgs, labels
        
    def __call__(self, imgs, labels):
        if random.random() > self.p:
            return imgs, labels
        if self.mode in ['horizontal']:
            imgs, labels = self.apply_horizontal(imgs, labels)
        return imgs, labels
    

class LitCls(LightningModule):

    def __init__(
            self,
            model: torch.nn.Module,
            learning_rate: float = 3e-4,
            cutmix_p: float = 0,
            cuts_num: int = 1,
    ) -> None:
        super().__init__()

        self.model: torch.nn.Module = model
        self.learning_rate: float = learning_rate
        self.aug_cutmix = CutMix(mode='horizontal', p=cutmix_p, cuts_num=cuts_num)
        
        self.loss: torch.nn.Module = nn.CrossEntropyLoss()
        metric_ce = MetricCollection({
            "CE": MeanMetric()
        })
        metric_auroc = MetricCollection({
            "AUROC": MultilabelAUROC(num_labels=182, average="macro"),
        })
        
        self.train_ce: MetricCollection = metric_ce.clone(prefix="train_")
        self.val_ce: MetricCollection = metric_ce.clone(prefix="val_")
        self.train_auroc: MetricCollection = metric_auroc.clone(prefix="train_")
        self.val_auroc: MetricCollection = metric_auroc.clone(prefix="val_")

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        x, y = self.aug_cutmix(x, y)
        preds = self.model(x)
        train_loss = self.loss(preds, y)
        self.train_ce(train_loss)
        self.log('train_loss', train_loss, prog_bar=True, sync_dist=True)
        self.train_auroc(F.sigmoid(preds), (y+0.9).int())
        return train_loss
    
    def on_train_epoch_end(self) -> None:
        self.log_dict(self.train_ce.compute(), sync_dist=True)
        self.train_ce.reset()
        self.log_dict(self.train_auroc.compute(), sync_dist=True)
        self.train_auroc.reset()

    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        preds = self.model(x)
        val_loss = self.loss(preds, y)
        self.val_ce(val_loss)
        self.val_auroc(F.sigmoid(preds), (y+0.9).int())

    def on_validation_epoch_end(self) -> None:
        self.log_dict(self.val_ce.compute(), prog_bar=True, sync_dist=True)
        self.val_ce.reset()
        self.log_dict(self.val_auroc.compute(), prog_bar=True, sync_dist=True)
        self.val_auroc.reset()

    def configure_optimizers(self):
        optimizer = AdamW(params=self.trainer.model.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return [optimizer], [scheduler]


if __name__ == "__main__":

    train_csv_path = 'tmp/train_noduplicates.csv'
    train_lab_path = 'tmp/train_prepared.csv'
    train_mels_path = 'tmp/train_mels/'
    train_stats_path = 'tmp/train_stats.csv'
    unlab_path = 'tmp/unlabeled_prepared.csv'

    # read train metadata
    data = pd.read_csv(train_lab_path)
    LABELS = sorted(data['primary_label'].unique())
    data['npy_data'] = list(data[['offset_seconds'] + LABELS].values)
    data.drop(columns=LABELS, inplace=True)
    data['mel_path'] = train_mels_path + data["filename"].apply(lambda s: str(s).replace('.ogg', '.npy'))

    # read unlabeled soundscapes metadata
    data_unlab = pd.read_csv(unlab_path)
    data_unlab['npy_data'] = list(data_unlab[['offset_seconds'] + LABELS].values)
    data_unlab = data_unlab.drop(columns=LABELS)

    # split only train_audio part
    if CFG.split_type == 'fold0':
        data_nodup = pd.read_csv(train_csv_path)
        kf_splitter = KFold(n_splits=5, random_state=42, shuffle=True)
        train_idx, valid_idx = list(kf_splitter.split(data_nodup, data_nodup['label']))[0]
        train_filenames = data_nodup.loc[train_idx].filename
        valid_filenames = data_nodup.loc[valid_idx].filename
    elif CFG.split_type == '80low':
        data_quantile = pd.read_csv(train_stats_path)
        train_filenames = data_quantile[data_quantile['T'] <= data_quantile['T_80q']].filename
        valid_filenames = data_quantile[data_quantile['T'] > data_quantile['T_80q']].filename
    train_df = data[data.filename.isin(train_filenames)]
    valid_df = data[data.filename.isin(valid_filenames)]

    train_df = train_df[train_df.good == 1]
    train_df = pd.concat([train_df, data_unlab])

    # keep only center chunks for validation
    valid_df = valid_df.groupby(['filename'], as_index=False).aggregate(func=lambda df: df.iloc[df.shape[0]//2])

    # create train and validation datasets
    transform = A.Compose([
    A.XYMasking(
        num_masks_x=(1, 12),
        num_masks_y=(1, 3),
        mask_x_length=(8, 16), 
        mask_y_length=(8, 16),
        fill_value=0,
        p=0.9,
        ),
    ])
    train_dataset = BirdDataset(df=train_df,
                                transform=transform, 
                                segm_sec=CFG.segmentation_sec)
    valid_dataset = BirdDataset(df=valid_df)
    datamodule = BirdDataModule(train_dataset, valid_dataset, batch_size=CFG.batch_size)

    # create pretrained model
    model = timm.create_model(
        CFG.model_name, pretrained=True,
        num_classes=182, in_chans=1,
    )
    lit_cls = LitCls(model, cutmix_p=0.9, learning_rate=CFG.learning_rate)

    # create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=None,  # save only last
        filename='{epoch}-{val_AUROC:.3f}',
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rich_progress = RichProgressBar()

    with open('/wandb_key.txt') as f:
        WANDB_KEY = f.readline()
    wandb.login(key=WANDB_KEY)
    logger = WandbLogger(
        project='BirdCLEF',
        log_model=True,
    )

    # create trainer and start training process
    trainer = Trainer(
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        max_epochs=CFG.epochs_num,
        accumulate_grad_batches=1,
        callbacks=[rich_progress, lr_monitor, checkpoint_callback],
        logger=logger,
        log_every_n_steps=50,
        accelerator='gpu',
    )
    trainer.fit(lit_cls, datamodule=datamodule)
    wandb.finish()
