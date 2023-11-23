import torch
from torch.utils.data import Dataset
import os
from typing import Callable
from glob import glob

from ml.utils.constants import DATA_DIR
from ml.utils.audio_util import AudioUtil

class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        audio_max_ms: int,
        mel_spectrogram: dict,
        transform: Callable=None,
    ) -> None:
        self.data_dir = data_dir
        self.file_list = glob(self.data_dir + os.sep + "*.wav")

        self.audio_max_ms = audio_max_ms
        self.mel_sg = mel_spectrogram
        self.transform = transform


    def __getitem__(self, index):
        file_path = self.file_list[index]
        label = int(file_path.split(os.sep)[-1].split('_')[0])

        audio = AudioUtil.open(file_path)
        audio = AudioUtil.pad_trunc(audio, self.audio_max_ms)
        # audio = AudioUtil.time_shift(audio, self.shift_pct)
    
        mel_spectro = AudioUtil.mel_spectrogram(
                                    audio, 
                                    n_mels = self.mel_sg['n_mels'], 
                                    n_fft = self.mel_sg['n_fft'], 
                                    win_len = self.mel_sg['win_length'],
                                    hop_len = self.mel_sg['hop_length'],
                                    f_min = self.mel_sg['f_min'],
                                    f_max = self.mel_sg['f_max'],
                                    pad = self.mel_sg['pad'],
                                    )

        mel_spectro_aug = self.transform(mel_spectro)
        return mel_spectro_aug, label

    def __len__(self):
        return len(self.file_list)
    
    def make_labels(self):
        return [file_path.split('_')[0] for file_path in self.file_list]