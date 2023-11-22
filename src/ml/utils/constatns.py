import os
from pathlib import Path
import torch
import yaml

ROOT_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = os.path.join(ROOT_DIR, "data")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, ".experiments")
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(CONFIG_DIR, 'config.yaml')) as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)