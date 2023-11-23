import torch
from torch.utils.data import Dataset
import os
from typing import Callable

from ml.utils.constants import DATA_DIR
from ml.utils.audio_util import AudioUtil

# DATA_DIR = os.path.join(DATA_DIR, 'raw')
# DATA_DIR = os.path.join(DATA_DIR, 'audio-mnist-whole')

class AudioDataset(Dataset):
    def __init__(
        self,
        data_dir,
        audio_max_len: int,
        mel_spectrogram: dict,
        transform: Callable=None,
    ) -> None:
        self.data_dir = data_dir
        self.file_list = os.listdir(data_dir)

        self.audio_max_len = audio_max_len
        self.mel_sg = mel_spectrogram
        self.transform = transform


    def __getitem__(self, index):
        file_name = self.file_list[index]
        label = file_name.split('_')[0]

        audio = AudioUtil.open(file_name)
        audio = AudioUtil.pad_trunc(audio, self.audio_max_len)
        # audio = AudioUtil.time_shift(audio, self.shift_pct)
        sgram = AudioUtil.mel_spectrogram(
                                    audio, 
                                    n_mels = self.mel_sg['n_mels'], 
                                    n_fft = self.mel_sg['n_fft'], 
                                    win_len = self.mel_sg['win_length'],
                                    hop_len = self.mel_sg['hop_length'],
                                    pad = self.mel_sg['pad'],
                                    f_min = self.mel_sg['f_min'],
                                    f_max = self.mel_sg['f_max']
                                    )
        # aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        # x = torch.load(os.path.join(self.data_dir, file_name))[0]

        return audio, label

    def __len__(self):
        return len(self.file_list)
    
    def make_labels(self):
        return [file_name.split('_')[0] for file_name in self.file_list]