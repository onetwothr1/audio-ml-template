import torch
from torch import nn
import lightning as L
from torchmetrics import F1Score

class LitModule(L.LightningModule):
    def __init__(
        self, 
        net: nn.Module, 
        loss_module: nn.Module, 
        num_classes: int,
        # metric_module: nn.Moudle,
    ) -> None:
        super().__init__()
        self.net = net
        self.loss_module = loss_module
        self.metric_module = F1score(task='multiclass', num_classes=6)
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x)

        loss = self.loss_module(pred, y)
        self.log("train/loss", loss.item())

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.net(x)

        loss = self.loss_module(pred, y)
        self.log("val/loss", loss.item(), on_epoch=True, on_step=False)

        acc = self.metric_module(pred, y)
        self.log('val/acc', on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass