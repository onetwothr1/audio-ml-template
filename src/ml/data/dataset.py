import os
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from typing import Callable
from glob import glob

class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        test: bool=False,
        df: pd.DataFrame=pd.DataFrame(),
        transform: Callable=None,
    ) -> None:
        self.data_dir = data_dir
        self.df = df
        self.transform = transform
        self.test = test

    def __getitem__(self, idx):
        file_path = self.df['file_path'].iloc[idx]
        x = torchaudio.load(self.data_dir + file_path)
        x = self.transform(x)  # augment, extract fetures (Mel-Spectrogram / MFCC / Deeplearing-based)
        if self.test:
            return x
        y = int(self.df['label'].iloc[idx])
        return x, y

    def __len__(self):
        return len(self.df)