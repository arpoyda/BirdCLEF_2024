import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

train_csv_path = 'tmp/train_noduplicates.csv'
train_path = 'data/train_audio/'
output_path = 'tmp/train_google_labeled.csv'


#=============================#
#== Prepairing Google model ==#
#=============================#

import tensorflow_hub as hub
model = hub.load('https://www.kaggle.com/models/google/bird-vocalization-classifier/TensorFlow2/bird-vocalization-classifier/1')
labels_path = hub.resolve('https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier/versions/1') + "/assets/label.csv"

# getting species from Google model
labels = pd.read_csv(labels_path)
google_species_to_label = {k: v for k, v in zip(labels.index, labels['ebird2021'])}
google_species_to_label_reverse = {k: v for k, v in zip(labels['ebird2021'], labels.index)}
species_google = set(list(google_species_to_label.values()))

# getting species from BirdCLEF 2024
train2024 = pd.read_csv(train_csv_path)
train2024['n_chunks'] = train2024['lengths'] // 160_000

species2024 = train2024['primary_label']
species2024 = set(list(species2024))

# species that are in BirdCLEF 2024 but no in Google model
new_species = list(species2024 - species_google)

# matching Google labels to 2024 by reindexing
train2024_google = train2024[~train2024['primary_label'].isin(new_species)]
species_2024_x_google = train2024_google['primary_label'].unique()

species_to_label_2024_x_google = {}
label_to_index_2024_x_google_reset_index = {}
for i, spec in enumerate(list(species_2024_x_google)):
    species_to_label_2024_x_google[spec] = google_species_to_label_reverse[spec]
    label_to_index_2024_x_google_reset_index[i] = spec
    
actual_indeces = np.array(list(species_to_label_2024_x_google.values()))


#=========================#
#== Labeling Train Audio ==#
#=========================#

google_probs = {}
for i in tqdm(range(len(train2024))):
    obj = train2024.iloc[i]
    filename = obj['filename']
    n_chunks = obj['n_chunks']
    length = obj['lengths']
    x_full, sr = librosa.load(train_path + obj['filename'], sr=32_000)
    x_full = x_full.astype(np.float32)
    step = sr * 5
    offset_probs = {}
    if n_chunks == 0:
        x_full = np.hstack([x for _ in range(int(5 / (x_full.shape[0] / sr)) + 1)])
        n_chunks = 1
    for i in range(n_chunks):
        x = x_full[step * i : step * (i+1)]
        logits = None
        if obj['primary_label'] in new_species:
            logits = np.zeros(len(actual_indeces)) - 20.0
        else:
            logits = model.infer_tf(x[np.newaxis, :])[0][0].numpy()[actual_indeces]
        offset_probs[i * 5] = list(logits)
    google_probs[filename] = offset_probs

filename_list = []
offset_list = []
logits_list = []
for filename, offset_dict in google_probs.items():
    for offset, logits in offset_dict.items():
        filename_list.append(filename)
        offset_list.append(offset)
        logits_list.append(logits)

logits_array = np.array(logits_list)

metainfo_df = pd.DataFrame({'filename': filename_list, 'offset_seconds': offset_list})
logit_df = pd.DataFrame(logits_array, columns=list(label_to_index_2024_x_google_reset_index.values()))
df = pd.concat([metainfo_df, logit_df], axis=1)
df = pd.merge(df, train2024, on=['filename'])

# adding new species to labels
for spec in new_species:
    df[spec] = -20.0
    df[spec][df['primary_label'] == spec] = 3.0

df.to_csv(output_path, index=False)
