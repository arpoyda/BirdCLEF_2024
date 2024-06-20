import numpy as np
import pandas as pd
import os
import librosa
import joblib
from tqdm import tqdm
    

train_csv_path = 'tmp/train_noduplicates.csv'
train_path = 'data/train_audio/'
train_output_path = 'tmp/train_mels/'


def mel(arr, sr=32_000):
    arr = arr * 1024
    spec = librosa.feature.melspectrogram(y=arr, sr=sr,
                                          n_fft=1024, hop_length=500, n_mels=128, 
                                          fmin=40, fmax=15000, power=2.0)
    spec = spec.astype('float32')
    return spec


print("Creating mel spectrograms from train_audio...")
data = pd.read_csv(train_csv_path)
data['path'] = train_path + data["filename"]
data['mel_path'] = train_output_path + data["filename"].apply(lambda s: s.replace('.ogg', '.npy'))

def process_train(idx):
    row = data.iloc[idx]
    audio, sr = librosa.load(row['path'], sr=32_000)
    mel_spec = mel(audio, sr)
    os.makedirs('/'.join(row['mel_path'].split('/')[:-1]), exist_ok=True)
    np.save(row['mel_path'], mel_spec)

indexes = data.index
_ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_train)(idx) for idx in tqdm(indexes, total=len(indexes))
    )
print("Done")
