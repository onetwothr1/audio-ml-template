import torch
from torch import nn
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

class HFAutoClassifier(torch.nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.model_name = kwargs.get('model_name')
        self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
        self.model.classifier = nn.Identity()
        self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x):
        output = self.model(x)
        output = self.classifier(output.logits)
        return output