import os
import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument('-d', '--dir', help='series name', dest='dir_name')
# parser.add_argument('--upscale', help='use upscaler gan model', dest='upscale', action='store_true', default=False)
# parser.add_argument('--high_color', help='increase size of video by 40 times', dest='high_color', action='store_true', default=False)

# args = {}
# for name, value in vars(parser.parse_args()).items():
#     args[name] = value

print('======= TRAIN PREPROCESSING =======')

print('--Dropping duplicates...')
os.system("python ./src/drop_duplicates.py")

print('--Computing mels...')
os.system("python ./src/compute_mels_train.py")

print('--Computing statistics...')
os.system("python ./src/compute_stats.py")

print('--Labeling by google...')
os.system("python ./src/train_google_labeling.py")

print('--Processing labels...')
os.system("python ./src/train_preproc_labels.py")