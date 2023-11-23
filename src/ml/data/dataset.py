import torch
from torch.utils.data import Dataset
import os

from ml.utils.constants import DATA_DIR

# DATA_DIR = os.path.join(DATA_DIR, 'raw')
# DATA_DIR = os.path.join(DATA_DIR, 'audio-mnist-whole')

class CustomDataset(Dataset):
    def __init__(
        self,
        data_dir,
        # gt_list,
    ) -> None:
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.labels = self.make_labels()

    def __getitem__(self, index):
        file_name = self.file_list[index]
        label = file_name.split('_')[0]
        x = torch.load(os.path.join(self.data_dir, file_name))[0]  
        return x, label

    def __len__(self):
        return len(self.file_list)
    
    def make_labels(self):
        return [file_name.split('_')[0] for file_name in self.file_list]