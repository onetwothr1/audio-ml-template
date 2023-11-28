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

# model = HFAutoClassifier(num_classes=6, **{'model_name': 'Rajaram1996/Hubert_emotion'})
# for name, _ in model.named_children():
#     print(name)
# num_layers = sum(1 for _ in model.model.hubert.encoder.layers.named_children())
# print("Number of layers:", num_layers)

# for name, param in model.encoder.named_parameters():
#     print(name)