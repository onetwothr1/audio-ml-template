import os
from pathlib import Path
import torch
import yaml

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = "/home/elicer/project/data/raw/audio-mnist-whole"
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")

with open(os.path.join(CONFIG_DIR, 'wandb_api_key.txt'), 'r') as file:
    WANDB_API_KEY = file.readline()

with open(os.path.join(CONFIG_DIR, 'config.yaml')) as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)