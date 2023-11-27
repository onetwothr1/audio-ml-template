from torch import nn

class BaseLine(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.num_classes = num_classes
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
        self.lin = nn.Linear(in_features=64, out_features=self.num_classes)
        if kwargs.get('he_initialization'):
            self._initialize_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.ap(x)
        x = x.view(x.shape[0], -1)
        return self.lin(x)

    def _initialize_weights(self):
        # He Initialization for convolutional layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # He Initialization for convolutional layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # He Initialization for fully connected layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
