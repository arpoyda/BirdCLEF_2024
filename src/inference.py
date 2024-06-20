import numpy as np
import pandas as pd
import os
import joblib
import openvino as ov
import librosa
import torch
import timm
import torch.nn.functional as F

def mel(arr, sr=32_000):
    arr = arr * 1024
    spec = librosa.feature.melspectrogram(y=arr, sr=sr,
                                          n_fft=1024, hop_length=500, n_mels=128, 
                                          fmin=40, fmax=15000, power=2.0)
    spec = spec.astype('float32')
    return spec

def torch_to_ov(model, input_shape=[48, 1, 128, 0]):
    core = ov.Core()
    ov_model = ov.convert_model(model)
    # ov_model.reshape(input_shape)
    compiled_model = core.compile_model(ov_model)
    return compiled_model

model_paths = [
    # EfficientNet_b0
    'models_weights/80low_trainnolab_10sec_gr20sec_ep7oo7.ckpt',
    'models_weights/train80low_nolab_10sec_gr60sec_lr1e3_ep12.ckpt',
    'models_weights/fold0_10sec_30secav_trainnolab_gr60sec_ep9.ckpt',
    # RegNetY
    'models_weights/regnety_10sec_gr60sec_fold0.ckpt',
    'models_weights/regnety_10sec_gr30sec_fold0.ckpt',
    'models_weights/regnety_10sec_gr30s_80low.ckpt',
]

test_audio_path = 'data/test_soundscapes/'
unlab_audio_path = 'data/unlabeled_soundscapes/'
train_metadata_path = 'data/train_metadata.csv'
submission_path = 'tmp/submission.csv'

SHAPE = [48, 1, 128, 320*2]

# load and compile models
models = []
for i, model_path in enumerate(model_paths):
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    model_name = 'efficientnet_b0'
    if 'regnety' in model_path:
        model_name = 'regnety_008.pycls_in1k'
    
    model = timm.create_model(
        model_name, pretrained=None,
        num_classes=182, in_chans=SHAPE[1]
    )
    
    new_state_dict = {}
    for key, val in state_dict['state_dict'].items():
        if key.startswith('model.'):
            new_state_dict[key[6:]] = val
    model.load_state_dict(new_state_dict)
    model.eval()
    model = torch_to_ov(model, input_shape=SHAPE)
    models.append(model)
print(f'{len(models)} models are ready')

# load and preprocess data
test_paths = [test_audio_path + f for f in sorted(os.listdir(test_audio_path))]
if len(test_paths) <= 1:
    test_paths = [unlab_audio_path + f for f in sorted(os.listdir(unlab_audio_path))][:2]
test_df = pd.DataFrame(test_paths, columns=['filepath'])
test_df['filename'] = test_df.filepath.map(lambda x: x.split('/')[-1].replace('.ogg',''))

def process(idx):
    row = test_df.iloc[idx]
    audiopath = row['filepath']
    audio, sr = librosa.load(audiopath, sr=None)
    chunk_size = sr * 5
    chunks = audio.reshape(-1, chunk_size)
    chunks_mel = mel(chunks, sr=sr)[:, :, :320].astype(np.float32)
    return row['filename'], chunks_mel
    
indexes = test_df.index
output = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process)(idx) for idx in indexes
    )
mels_dict = dict(output)

# make predictions
ids = []
preds = [np.empty(shape=(0, 182), dtype='float32') for _ in range(len(models))]
with torch.no_grad():
    for filename in test_df.filename.tolist():
        chunks = mels_dict[filename]
        chunks = chunks[:, np.newaxis, :, :]
        chunks_1 = np.concatenate([chunks[:1], chunks[:-1]], axis=0)
        chunks_2 = np.concatenate([chunks[1:], chunks[-1:]], axis=0)
        chunks = np.concatenate([chunks_1, chunks, chunks_2], axis=-1)
        chunks = chunks[...,160:-160]
        chunks = librosa.power_to_db(chunks, ref=1, top_db=100.0).astype('float32')
        chunks = torch.from_numpy(chunks)
        for m_idx in range(len(models)):
            rec_preds = models[m_idx](chunks)[0]
            preds[m_idx] = np.concatenate([preds[m_idx], rec_preds], axis=0)
        # create ID for each chunk in the audio with the filename and frame number
        rec_ids = [f'{filename}_{(frame_id+1)*5}' for frame_id in range(rec_preds.shape[0])]
        ids += rec_ids

# postprocessing and ensembling
preds = F.sigmoid(torch.Tensor(np.array(preds))).numpy()
s0 = preds.shape[0]
preds = preds.reshape(s0, -1, 48, 182)
smooth_preds = preds.copy()
for i in range(48):
    smooth_preds[:, :, i] = preds[:, :, max(0,i-2):i+3].mean(axis=-2)
preds = smooth_preds.reshape(s0, -1, 182)
preds = preds.mean(axis=0, keepdims=True)
preds = preds.squeeze()

# create submission file
data = pd.read_csv(train_metadata_path)
LABELS = sorted(list(data['primary_label'].unique()))
pred_df = pd.DataFrame(ids, columns=['row_id'])
pred_df.loc[:, LABELS] = preds
pred_df.to_csv(submission_path, index=False)
