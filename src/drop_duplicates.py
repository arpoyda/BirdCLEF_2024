import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


train_csv_path = 'data/train_metadata.csv'
train_path = 'data/train_audio/'
output_path = 'tmp/train_noduplicates.csv'


def get_length(f):
    x, _ = librosa.load(f, sr=32_000)
    return x.shape[0]


train2024 = pd.read_csv(train_csv_path)

lengths = joblib.Parallel(n_jobs=-1, backend="loky")(
    joblib.delayed(get_length)(f) for f in tqdm(train_path + train2024['filename'], total=len(train2024['filename']))
)

train2024['lengths'] = lengths
train2024 = train2024.drop_duplicates(subset=['primary_label', 'author', 'lengths'])
train2024.to_csv(output_path, index=False)

