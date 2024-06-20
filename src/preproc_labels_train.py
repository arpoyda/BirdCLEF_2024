import numpy as np
import pandas as pd
from ast import literal_eval
import torch
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


train_csv_path = 'tmp/train_google_labeled.csv'
output_path = 'tmp/train_prepared.csv'

data = pd.read_csv(train_csv_path, converters={'secondary_labels': literal_eval})
LABELS = sorted(data['primary_label'].unique())

# find 'good' chunks (predicted label == ground truth label)
data['predict'] = data[LABELS].idxmax(axis=1)
wrong_idxs = data[data['predict'] != data['primary_label']].index
right_idxs = data[data['predict'] == data['primary_label']].index
sec_idxs = list()
for idx in wrong_idxs:
    row = data.iloc[idx]
    if row.predict in row.secondary_labels:
        sec_idxs.append(idx)
        data.loc[idx, 'primary_label'] = row.predict
        data.at[idx, 'secondary_labels'] = list()
right_idxs = right_idxs.tolist() + sec_idxs
all_wrong_filenames = (set(data.loc[wrong_idxs].filename.unique()) - set(data.loc[right_idxs].filename.unique()))
data['good'] = 0
data.loc[right_idxs, 'good'] = 1
data.loc[data.filename.isin(all_wrong_filenames), 'good'] = 1

data[LABELS] = F.softmax(torch.Tensor(data[LABELS].values), dim=-1).numpy()
pseudo_labels = data[LABELS].values
ohe = OneHotEncoder(categories=[np.array(range(182))], sparse_output=False)
true_labels = ohe.fit_transform(data['label'].values.reshape(-1, 1))
data[LABELS] = true_labels

# add secondary labels
idxseclab = data[data.secondary_labels.apply(lambda s: len(s) > 0)].index
for idx in idxseclab:
    sec_labs = data.loc[idx, 'secondary_labels']
    sec_labs = list(set(sec_labs) & set(LABELS))
    if len(sec_labs) > 0:
        prime_lab = data.loc[idx, 'primary_label']
        data.loc[idx, prime_lab] = 0.5
        data.loc[idx, sec_labs] = 0.5 / len(sec_labs)
true_labels = data[LABELS].values
alpha = 0.95
labels = alpha*true_labels + (1-alpha)*pseudo_labels
data[LABELS] = labels

data.to_csv(output_path, index=False)
