import torch
from torch.utils.data import Dataset
import os
from typing import Callable
from glob import glob

from util.constants import DATA_DIR

class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform: Callable=None,
    ) -> None:
        self.data_dir = data_dir
        self.file_list = glob(self.data_dir + os.sep + "*.wav")
        self.transform = transform

    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = int(file_path.split(os.sep)[-1].split('_')[0])

        # laod .wav file, augment, extract fetures (Mel-Spectrogram / MFCC / Deeplearing-based)
        data = self.transform(file_path)
        return data, label

    def __len__(self):
        return len(self.file_list)
    
    def make_labels(self):
        return [file_path.split('_')[0] for file_path in self.file_list]