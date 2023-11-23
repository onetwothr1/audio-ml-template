import torch
from torch import nn
from torch import optim
from torchmetrics import F1Score
import numpy as np
import lightning as L

class LitModule(L.LightningModule):
    def __init__(
        self, 
        net: nn.Module, 
        loss_module: nn.Module, 
        num_classes: int,
        optim_config: dict,
        lr_schdlr_config: dict
    ) -> None:
        super().__init__()
        self.net = net
        if loss_module=='CrossEntropyLoss':
            self.loss_module = nn.CrossEntropyLoss()
        self.metric_module = F1Score(task='multiclass', num_classes=num_classes)
        self.optim_config = optim_config
        self.lr_schdlr_config = lr_schdlr_config

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

    def configure_optimizers(self):
        if self.optim_config['class_path']=='SGD':
            optimizer = optim.SGD(
                                params = self.net.parameters(), 
                                lr = self.optim_config['init_args']['lr'],
                                momentum = self.optim_config['init_args']['momentum'],
                                weight_decay = self.optim_config['init_args']['weight_decay'])
        lr_scheduler = None
        if self.lr_schdlr_config:
            return [optimizer], [lr_scheduler]
        return optimizer

