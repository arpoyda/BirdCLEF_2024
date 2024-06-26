import numpy as np
import pandas as pd
import os
import librosa
import joblib
from tqdm import tqdm
    

unlab_path = 'data/unlabeled_soundscapes/'
unlab_output_path = 'tmp/unlabeled_mels/'


def mel(arr, sr=32_000):
    arr = arr * 1024
    spec = librosa.feature.melspectrogram(y=arr, sr=sr,
                                          n_fft=1024, hop_length=500, n_mels=128, 
                                          fmin=40, fmax=15000, power=2.0)
    spec = spec.astype('float32')
    return spec


print("Creating mel spectrograms from unlabeled_soundscapes...")
paths = [unlab_path + f for f in sorted(os.listdir(unlab_path))]
data_unlabeled = pd.DataFrame(paths, columns=['filepath'])
data_unlabeled['filename'] = data_unlabeled.filepath.map(lambda x: x.split('/')[-1])
data_unlabeled['mel_path'] = unlab_output_path + data_unlabeled["filename"].apply(lambda s: s.replace('.ogg', '.npy'))

def process_unlabeled(idx):
    row = data_unlabeled.iloc[idx]
    audio, sr = librosa.load(row['filepath'], sr=None)
    mel_spec = mel(audio, sr=sr)
    os.makedirs('/'.join(row['mel_path'].split('/')[:-1]), exist_ok=True)
    # saving 1 minute per file to speed up reading during training
    for i in range(4):
        start = i * 320 * 12
        end = (i + 1) * 320 * 12
        np.save(row['mel_path'].replace('.npy', f'_{i}.npy'), mel_spec[:, start:end])

indexes = data_unlabeled.index
_ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_unlabeled)(idx) for idx in tqdm(indexes, total=len(indexes))
    )
print("Done")
