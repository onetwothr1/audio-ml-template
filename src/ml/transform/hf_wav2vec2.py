import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2FeatureExtractor

from .base import Transform


class Wav2Vec2Extractor(Transform):
    def __init__(self, train:bool=True, **kwargs):
        self.train = train
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
        if self.train:
            x, y = zip(*batch)
            x = pad_sequence(x).mT
            y = pad_sequence(torch.tensor(y).unsqueeze(1)).mT.squeeze(-1)
            return x, y
        else:
            x = pad_sequence(batch).mT
            return x


# model = Wav2Vec2Extractor(num_classes=6, **{'model_name': 'Rajaram1996/Hubert_emotion',
#                                             'sample_rate': 16000, 'audio_max_ms': 4000})
# for name, param in model.extractor().named_parameters():
#     print(name)