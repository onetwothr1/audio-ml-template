import torch
from torch import nn

class BaseLine(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential([
            nn.Linear(40,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,100),
            nn.ReLU(),
            nn.Linear(100,10),
            nn.Softmax()
        ])

    def forward(self, x):
        return self.net(x)