import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2FeatureExtractor

from .base import Transform


class Wav2Vec2Extractor(Transform):
    def __init__(self, **kwargs):
        self.model_name = kwargs.get('model_name')
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)

        self.sampling_rate = kwargs.get('sample_rate')
        self.max_length = kwargs.get('audio_max_ms')

    def train_transform(self, x):
        audio = x[0]
        feature = self.extractor(
                    audio,
                    sampling_rate=self.sampling_rate,
                    max_length=self.max_length,
                    padding=True, # DO NOT CHANGE THIS VALUE! FATAL ERROR RAISES!
                    truncation=True,
                    return_tensors='pt').input_values
        return feature.squeeze()

    def val_transform(self, x):
        return self.train_transform(x)

    def test_transform(self, x):
        return self.val_transform(x)
    
    def collate_fn(self, batch):
        x, y = zip(*batch)
        # x.permute(*torch.arange(x.ndim - 1, -1, -1)) <- it takes transpose of x. that's all
        # this code does not make training slower.
        x = pad_sequence([xi.permute(*torch.arange(xi.ndim - 1, -1, -1)) for xi in x], batch_first=True) # get transpose of x
        y = pad_sequence([torch.tensor([yi]).permute(*torch.arange(torch.tensor([yi]).ndim - 1, -1, -1)) for yi in y], batch_first=True).squeeze(-1)  # Convert scalar targets to 1D tensors
        return x, y
