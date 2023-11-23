from torch import nn

class BaseLine(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5,5), stride=(2,2), padding=(2,2)),
            nn.ReLU(),
            nn.BatchNorm2d(8),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            )
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        return self.lin(x)