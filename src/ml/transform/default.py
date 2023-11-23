import numpy as np
import torch
from torchaudio import transforms

from ml.transform.base import BaseTransforms

class Transform(BaseTransforms):
    def __init__(
        self,
    ) -> None:
        super().__init__()
        self.do_time_shift = True
        self.do_masking = True
        self.do_noising = True

    def time_shift(spec, shift_max: float):
        time_len = spec.size(2)
        shift_amt = int(random.random() * shift_max * time_len)
        return torch.roll(spec, shifts=shift_amt, dims=2)

    def masking(self, spec: torch.tensor, max_mask_pct: float=0.1, n_freq_mask: int=1, n_time_mask: int=1):
        n_mels, n_steps = data.size()
        masking_value = data.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_mask):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_mask):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)
        return aug_spec

    def noising(self, data: torch.tensor, noise_level: float):
        noise = torch.randn_like(data) * noise_level
        aug_data = data + noise
        return aug_data

    def train_transform(self, x):
        if self.do_time_shift:
            x = self.time_shift(x)
        if self.do_masking(x):
            x = self.masking(x)
        if self.do_noising(x):
            x = self.noising(x, noise_level)
        return x

    def val_transform(self, x):
        return x
    
    def test_transform(self, x):
        return x