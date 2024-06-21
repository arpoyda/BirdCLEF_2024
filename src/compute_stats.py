import numpy as np
import pandas as pd
import librosa
import joblib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


train_csv_path = 'tmp/train_noduplicates.csv'
train_path = 'data/train_audio/'
output_path = 'tmp/train_stats.csv'


def calculate_statistics(x):
    std = np.nanstd(x)
    var = np.nanvar(x)
    rms = np.nanmean(np.sqrt(x**2))
    pwr = np.nanmean(x**2)
    return np.hstack([std, var, rms, pwr])

train2024 = pd.read_csv(train_csv_path)
train2024['path'] = train_path + train2024["filename"]

def get_stats(i):
    p = train2024.iloc[i]['path']
    x, _ = librosa.load(p, sr=32_000)
    x = x[: 60 * 32_000]
    return calculate_statistics(x)

stats = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(get_stats)(i) for i in tqdm(range(len(train2024)))
    )
    
stats = np.array(stats)
stats = pd.DataFrame(stats[:, :], columns=['std', 'var', 'rms', 'pwr'])
train2024 = pd.concat([train2024, stats], axis=1)

train2024[['std_s', 'var_s', 'rms_s', 'pwr_s']] = StandardScaler().fit_transform(train2024[['std', 'var', 'rms', 'pwr']])
train2024['T'] = train2024['std_s'] + train2024['var_s'] + train2024['rms_s'] + train2024['pwr_s']

quantile = train2024.groupby('primary_label')['T'].quantile(0.8).reset_index()
quantile.columns = ['primary_label', 'T_80q']
data24_cp = pd.merge(train2024, quantile, on='primary_label')
data24_cp.to_csv(output_path, index=False)
