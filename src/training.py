import numpy as np
import pandas as pd
import json
import warnings
import argparse

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=pd.errors.SettingWithCopyWarning)

from sklearn.model_selection import KFold

from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichProgressBar

import timm
import albumentations as A
# import wandb

from training_utils import BirdDataset, BirdDataModule, LitCls


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-с', '--cfg', help='Model config file path', dest='cfg_path')
    args = {}
    for name, value in vars(parser.parse_args()).items():
        args[name] = value


    train_csv_path = 'tmp/train_noduplicates.csv'
    train_lab_path = 'tmp/train_prepared.csv'
    train_mels_path = 'tmp/train_mels/'
    train_stats_path = 'tmp/train_stats.csv'
    unlab_path = 'tmp/unlabeled_prepared.csv'
    unlab_mels_path = 'tmp/unlabeled_mels/'
    models_path = 'models_weights/'

    # load config
    with open(args['cfg_path'], 'r') as f:
        CFG = json.load(f)

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
    data_unlab['mel_path'] = unlab_mels_path + data_unlab["filename"]

    # split only train_audio part
    if CFG['split_type'] == 'fold0':
        data_nodup = pd.read_csv(train_csv_path)
        kf_splitter = KFold(n_splits=5, random_state=42, shuffle=True)
        train_idx, valid_idx = list(kf_splitter.split(data_nodup, data_nodup['primary_label']))[0]
        train_filenames = data_nodup.loc[train_idx].filename
        valid_filenames = data_nodup.loc[valid_idx].filename
    elif CFG['split_type'] == '80low':
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
                                segm_sec=CFG['segmentation_sec'])
    valid_dataset = BirdDataset(df=valid_df)
    datamodule = BirdDataModule(train_dataset, train_dataset, 
                                batch_size=CFG['batch_size'])

    # create pretrained model
    model = timm.create_model(
        CFG['model_name'], pretrained=True,
        num_classes=182, in_chans=1,
    )
    lit_cls = LitCls(model, cutmix_p=0.9, learning_rate=CFG['learning_rate'])

    # create callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=None,  # save only last
        filename='{epoch}-{val_AUROC:.3f}',
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    rich_progress = RichProgressBar()

    # with open('/wandb_key.txt') as f:
    #     WANDB_KEY = f.readline()
    # wandb.login(key=WANDB_KEY)
    # logger = WandbLogger(
    #     project='BirdCLEF',
    #     log_model=True,
    # )

    # create trainer and start training process
    trainer = Trainer(
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        max_epochs=CFG['epochs_num'],
        accumulate_grad_batches=1,
        callbacks=[rich_progress, lr_monitor, checkpoint_callback],
        # logger=logger,
        log_every_n_steps=50,
        # accelerator='gpu',
    )
    trainer.fit(lit_cls, datamodule=datamodule)
    # wandb.finish()

    name = args['cfg_path'].split('/')[-1].split('.')[0]
    trainer.save_checkpoint(f'{models_path}{CFG["name"]}.ckpt')
    