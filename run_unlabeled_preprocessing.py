import os

print('======= UNLABELED PREPROCESSING =======')

print('--Computing mels...')
os.system("python ./src/compute_mels_unlabeled.py")

print('--Labeling by google...')
os.system("python ./src/labeling_unlabeled_google.py")

print('--Labeling by EfficientNet...')
os.system("python ./src/labeling_unlabeled_effnet.py")

print('--Processing labels...')
os.system("python ./src/preproc_labels_unlabeled.py")
