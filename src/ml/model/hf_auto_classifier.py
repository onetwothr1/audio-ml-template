import torch
from torch import nn
from transformers import AutoModelForAudioClassification, Wav2Vec2FeatureExtractor


MODEL_NAME = "facebook/wav2vec2-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)

class HFAutoClassifier(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model_name = kwargs.get('model_name')
        self.model = AutoModelForAudioClassification.from_pretrained(MODEL_NAME)
        self.model.classifier = nn.Identity()
        self.num_classes = kwargs.get('num_classes')
        self.classifier = nn.Linear(256, self.num_classes)

    def forward(self, x):
        output = self.model(x)
        output = self.classifier(output.logits)
        return output