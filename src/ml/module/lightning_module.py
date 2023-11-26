import torch
from torch import nn
from torch import optim
from torchmetrics import F1Score
from torch.optim import lr_scheduler
import numpy as np
import lightning as L

class LitModule(L.LightningModule):
    def __init__(
        self, 
        net: nn.Module, 
        loss_module: nn.Module, 
        num_classes: int,
        lr: float,
        optim: dict,
        lr_scheduler: dict,
    ) -> None:
        super().__init__()
        self.net = net
        if loss_module=='CrossEntropyLoss':
            self.loss_module = nn.CrossEntropyLoss()
        self.lr = lr        
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.metric_module = F1Score(task='multiclass', num_classes=num_classes)

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = self.loss_module(pred, y)
        self.log("train/loss", loss.item())

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = self.loss_module(pred, y)
        self.log("val/loss", loss.item(), on_epoch=True, on_step=False)

        acc = self.metric_module(pred, y)
        self.log('val/acc', acc, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        acc = self.metric_module(pred, y)
        self.log("test/acc", acc)

    def configure_optimizers(self):
        if self.optim['class_path']=='SGD':
            optimizer = optim.SGD(
                                params = self.net.parameters(), 
                                lr = self.lr,
                                momentum = self.optim['init_args']['momentum'],
                                weight_decay = self.optim['init_args']['weight_decay'])
        if self.lr_scheduler['class_path']=='CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                                optimizer, 
                                T_max=self.lr_scheduler['init_args']['T_max'], 
                                eta_min = 1e-6)
        return [optimizer], [scheduler]