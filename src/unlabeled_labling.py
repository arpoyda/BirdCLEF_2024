import os
import argparse
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import librosa

import warnings
warnings.filterwarnings("ignore")
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

train_csv_path = 'data/train_metadata.csv'
unlabeled_path = 'data/unlabeled_soundscapes/'
output_path = 'tmp/labeled_soundscapes.csv'


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


#===================================#
#== Labling Unlabeled Soundscapes ==#
#===================================#

unlabeled_soundscapes = sorted(os.listdir(unlabeled_path))[:10]

paths = []
offsets = []
logits = []
for path in tqdm(unlabeled_soundscapes):
    n_chunks = 4 * 60 // 5
    x_full, sr = librosa.load(unlabeled_path + path, sr=32_000)
    if len(x_full) < 4 * 60 * sr:
        continue
    x_full = x_full.astype(np.float32)
    step = sr * 5
    for i in range(n_chunks):
        x = x_full[step * i : step * (i+1)]
        logits.append(model.infer_tf(x[np.newaxis, :])[0][0].numpy()[actual_indeces])
        offsets.append(i * 5)
        paths.append(path)

df = pd.DataFrame(np.array(logits).astype(np.float32), columns=list(label_to_index_2024_x_google_reset_index.values()))
df.insert(loc=0, column='offset_seconds', value=offsets)
df.insert(loc=0, column='filename', value=paths)
df.to_csv(output_path, index=False)