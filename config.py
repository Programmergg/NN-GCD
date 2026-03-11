# -----------------
# DATASET ROOTS
# -----------------
import os
DATASET_DIR = '/dataset'


cifar_10_root = f'{DATASET_DIR}/cifar10'
cifar_100_root = f'{DATASET_DIR}/cifar100'
cub_root =f'{DATASET_DIR}/cub'
aircraft_root = '/home/YS/SimGCD_empssl/dataset/fgvc-aircraft-2013b'
car_root = f'{DATASET_DIR}/cars'
herbarium_dataroot = f'{DATASET_DIR}/herbarium_19'
imagenet_root = f'{DATASET_DIR}/ImageNet100/images'

# OSR Split dir
osr_split_dir = 'data/ssb_splits'

# -----------------
# OTHER PATHS
# -----------------
exp_root = 'dev_outputs' # All logs and checkpoints will be saved here