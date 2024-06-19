import numpy as np
import pandas as pd
import librosa
import torch
import timm
from tqdm.notebook import tqdm

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

train_csv_path = 'data/train_metadata.csv'
unlab_path = 'tmp/labeled_google_soundscapes.csv'
unlab_mel_dir = 'tmp/unlabeled_mels/'
model_paths = [f'models_weights/fold{i}.ckpt' for i in [0, 1, 3]]
output_path = 'tmp/labeled_effnet_soundscapes.csv'

#=========================#
#== Prepairing Metadata ==#
#=========================#

data = pd.read_csv(train_csv_path)
LABELS = sorted(data['primary_label'].unique())
data_unlab = pd.read_csv(unlab_path)
data_unlab['offset_seconds'] = data_unlab['offset_seconds'].astype(int)
data_unlab['filename'] = data_unlab.filename.apply(lambda s: s.replace('.ogg', '')) + \
                             '_' + (data_unlab.offset_seconds // 60).astype(str) + '.npy'
data_unlab['mel_path'] = unlab_mel_dir + data_unlab.filename
data_unlab['offset_seconds'] = data_unlab['offset_seconds'] % 60
data_unlab.drop(columns=LABELS, inplace=True, errors="ignore")

#====================================#
#== Prepairing EfficientNet models ==#
#====================================#

global_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = []
for model_path in model_paths:
    state_dict = torch.load(model_path, map_location=global_device)
    model = timm.create_model(
        'efficientnet_b0', pretrained=None,
        num_classes=182, in_chans=1,
    )
    new_state_dict = {}
    for key, val in state_dict['state_dict'].items():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = val
    model.load_state_dict(new_state_dict)
    model.eval().to(global_device)
    models.append(model)
print(f'{len(models)} models are ready')

#====================================#
#== Labeling Unlabeled Soundscapes ==#
#====================================#

all_preds = np.empty((0, 182))
with torch.no_grad():
    for mel_path in tqdm(data_unlab.mel_path.unique()):
        spec = np.load(mel_path)
        pad_len = 320 - spec.shape[-1] % 320
        if pad_len >= 319:
            spec = spec[..., :spec.shape[-1] - spec.shape[-1] % 320]
        else:
            spec = np.pad(spec, ((0, 0), (0, pad_len)))
        spec = np.transpose(spec.reshape(128, -1, 320), axes=(1, 0, 2))
        spec = librosa.power_to_db(spec, ref=1, top_db=100.0)
        spec = torch.from_numpy(spec).unsqueeze(1).to(global_device)
        
        preds = np.zeros((spec.shape[0], 182))
        for model in models:
            preds += model(spec).cpu().numpy().astype('float32')
        preds /= len(models)
        all_preds = np.concatenate([all_preds, preds], axis=0)
data_unlab.loc[:, LABELS] = all_preds

data_unlab.to_csv(output_path, index=False)
