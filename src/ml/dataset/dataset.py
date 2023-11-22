import torch
from torch.utils.data import Dataset
import os

from ml.utils.constants import DATA_DIR

DATA_DIR = os.path.join(DATA_DIR, 'raw')
DATA_DIR = os.path.join(DATA_DIR, 'audio-mnist-whole')

class CustomDataset(Dataset):
    def __init__(self,
    data_dir,
    gt_list,
    augmentation
    ) -> None:
        self.file_list = os.listdir(data_dir)
        self.augmentation = augmentation

    def __getitem__(self, index):
        file_name = self.file_list[index]
        label = file_name.split('_')[0]
        x = torch.load(file_name)

        if self.augmentation:
            pass
        
        return x, label

    def __len__(self):
        return len(self.file_list)