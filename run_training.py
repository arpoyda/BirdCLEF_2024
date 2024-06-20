import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-с', '--cfg', help='Model config file path', dest='cfg_path')
args = {}
for name, value in vars(parser.parse_args()).items():
    args[name] = value


print('======= TRAINING =======')
os.system(f"python ./src/training.py --cfg {args['cfg_path']}")
