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
        self.file_list = glob(self.data_dir + os.sep + "*.wav")
        self.df = df
        # self.setup() # if df isn't given, this method will make appropriate DataFrame
        self.transform = transform
        self.test = test

    def __getitem__(self, idx):
        file_path = self.df['file_path'].iloc[idx]
        x = torchaudio.load('/home/elicer/project/월간 데이콘 음성 감정 인식 AI 경진대회' + file_path[1:])
        x = self.transform(x)  # augment, extract fetures (Mel-Spectrogram / MFCC / Deeplearing-based)
        if self.test:
            return x
        y = int(self.df['label'].iloc[idx])
        return x, y

    def __len__(self):
        return len(self.df)

    # def get_labels(self):
    #     return self.df['label']

    # def set_labels(self, labels):
    #     self.df['label'] = labels
    
    # def make_labels(self):
    #     return [file_path.split('_')[0] for file_path in self.file_list]

    # def setup(self):
    #     file_path_list = []
    #     label_list = []
    #     for file_path in self.file_list:
    #         file_path_list.append(file_path)
    #         label_list.append(int(file_path.split(os.sep)[-1].split('_')[0]))
    #     self.df['file_path'] = file_path_list
    #     self.df['label'] = label_list