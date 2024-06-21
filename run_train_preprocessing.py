import os

print('======= TRAIN PREPROCESSING =======')

print('--Dropping duplicates...')
os.system("python ./src/drop_duplicates.py")

print('--Computing mels...')
os.system("python ./src/compute_mels_train.py")

print('--Computing statistics...')
os.system("python ./src/compute_stats.py")

print('--Labeling by google...')
os.system("python ./src/labeling_train_google.py")

print('--Processing labels...')
os.system("python ./src/preproc_labels_train.py")
