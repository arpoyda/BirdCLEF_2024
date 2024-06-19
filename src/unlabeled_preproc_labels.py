import pandas as pd
import torch
import torch.nn.functional as F

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


train_csv_path = 'tmp/train_google_labeled.csv'
lab_google_path = 'tmp/labeled_google_soundscapes.csv'
lab_effnet_path = 'tmp/labeled_effnet_soundscapes.csv'
unlab_mel_dir = 'tmp/unlabeled_mels/'
output_path = 'tmp/unlabeled_prepared.csv'

data = pd.read_csv(train_csv_path)
LABELS = sorted(data['primary_label'].unique())

# load google labels
data_unlab_ggl = pd.read_csv(lab_google_path)
data_unlab_ggl[['bkrfla1', 'indrol2']] = -20.
data_unlab_ggl[LABELS] = F.softmax(torch.Tensor(data_unlab_ggl[LABELS].values), dim=-1).numpy()

# load efficientnet labels
data_unlab_effnet = pd.read_csv(lab_effnet_path)
data_unlab_effnet[LABELS] = F.softmax(torch.Tensor(data_unlab_effnet[LABELS].values), dim=-1).numpy()

# ensemble google and efficientnet labels
data_unlab = data_unlab_effnet.copy()
data_unlab[LABELS] = (data_unlab_ggl[LABELS].values + data_unlab_effnet[LABELS].values) / 2
data_unlab['primary_label'] = data_unlab[LABELS].idxmax(axis=1)

data_unlab.to_csv(output_path, index=False)
