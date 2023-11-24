import random
import numpy as np
import torch
import torchaudio
from torchaudio import transforms

from .base import Transform

'''
class for loading, processing, augmenting audio file
'''
class CustomTransform(Transform):
    def __init__(
        self,
        **kwargs,
        # audio_max_ms: int,
        # sample_rate: int,
        # mel_spectrogram: dict,
        # time_shift: dict,
        # masking: dict,
        # noising: dict
    ) -> None:
        super().__init__()

        self.audio_max_ms = kwargs.get('audio_max_ms')
        self.sample_rate = kwargs.get('sample_rate')
        self.mel_sg_cfg = kwargs.get('mel_spectrogram')
        self.time_shift_cfg = kwargs.get('time_shift')
        self.masking_cfg = kwargs.get('masking')
        self.noising_cfg = kwargs.get('noising')

    def train_transform(self, file_path):
        aud = Transform.open(file_path)
        aud = self.resample(aud)
        aud = self.pad_trunc(aud)

        if self.time_shift_cfg['use']:
            aud = self.time_shift(aud)
        if self.noising_cfg['use']:
            aud = self.noising(aud)

        spec = self.mel_spectrogram(aud)
        if self.masking_cfg:
            spec = self.masking(spec)
        return spec

    def val_transform(self, file_path):
        aud = Transform.open(file_path)
        aud = self.resample(aud)
        aud = self.pad_trunc(aud)
        spec = self.mel_spectrogram(aud)
        return spec
    
    def test_transform(self, file_path):
        return self.val_transform(file_path)


    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # Nothing to do
            return aud

        if (new_channel == 1):
            # Convert from stereo to mono by selecting only the first channel
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the first channel
            resig = torch.cat([sig, sig])

        return ((resig, sr))

    def resample(self, aud):
        sig, sr = aud

        if (sr == self.sample_rate):
            return aud

        num_channels = sig.shape[0]
        # Resample first channel
        resig = torchaudio.transforms.Resample(sr, self.sample_rate)(sig[:1,:])
        if (num_channels > 1):
            # Resample the second channel and merge both channels
            retwo = torchaudio.transforms.Resample(sr, self.sample_rate)(sig[1:,:])
            resig = torch.cat([resig, retwo])

        return ((resig, self.sample_rate))

    def pad_trunc(self, aud):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr//1000 * self.audio_max_ms

        if (sig_len > max_len):
            # Truncate the signal to the given length
            sig = sig[:,:max_len]

        elif (sig_len < max_len):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # Pad with 0s
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)
        
        return (sig, sr)
    
    def mel_spectrogram(self, aud):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(
                    sample_rate=sr,
                    n_mels = self.mel_sg_cfg['n_mels'], 
                    n_fft = self.mel_sg_cfg['n_fft'], 
                    win_length = self.mel_sg_cfg['win_length'],
                    hop_length = self.mel_sg_cfg['hop_length'],
                    f_min = self.mel_sg_cfg['f_min'],
                    f_max = self.mel_sg_cfg['f_max'],
                    pad = self.mel_sg_cfg['pad'],
                )(sig)


        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)


    def time_shift(self, spec):
        time_len = spec.size(2)
        shift_amt = int(random.random() * self.time_shift_cfg['shift_max'] * time_len)
        return torch.roll(spec, shifts=shift_amt, dims=2)

    def masking(self, spec: torch.tensor):
        n_mels, n_steps = spec.size(1), spec.size(2)
        masking_value = spec.mean()
        # aug_spec = spec
        max_mask_pct = self.masking_cfg['max_mask_percent']

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(self.masking_cfg['n_freq_mask']):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(spec, masking_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(self.masking_cfg['n_time_mask']):
            aug_spec = transforms.TimeMasking(time_mask_param)(spec, masking_value)
        return aug_spec

    def noising(self, aud: torch.tensor):
        sig, sr = aud
        noise = torch.randn_like(sig) * self.noising_cfg['noise_level']
        aug_data = sig + noise
        return aug_data, sr