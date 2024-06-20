import os

print('======= UNLABELED PREPROCESSING =======')

print('--Computing mels...')
os.system("python ./src/compute_mels_unlabeled.py")

print('--Labeling by EfficientNet...')
os.system("python ./src/unlabeled_effnet_labeling.py")

print('--Labeling by google...')
os.system("python ./src/unlabeled_google_labeling.py")

print('--Processing labels...')
os.system("python ./src/unlabeled_preproc_labels.py")