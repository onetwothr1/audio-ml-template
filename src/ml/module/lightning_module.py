import torch
from torch import nn
from torch import optim
from torch.optim import SGD
from torchmetrics import F1Score, Accuracy
from torch.optim import lr_scheduler
import numpy as np
import pandas as pd
import lightning as L
from typing import Union
import copy

from util.sam import SAM

class LitModule(L.LightningModule):
    def __init__(
        self, 
        net: nn.Module, 
        loss_module: nn.Module, 
        num_classes: int,
        lr: float,
        lr_layer_decay: Union[float,int],
        optim: dict,
        lr_scheduler: dict,
        last_epoch:int=-1
    ) -> None:
        super().__init__()
        self.net = net
        if loss_module=='CrossEntropyLoss':
            self.loss_module = nn.CrossEntropyLoss()
        self.lr = float(lr)        
        self.lr_layer_decay = float(lr_layer_decay)
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.last_epoch = last_epoch
        self.accuracy =  Accuracy(task='multiclass', num_classes=num_classes)
        self.f1score = F1Score(task='multiclass', num_classes=num_classes)
        self.test_preds = []

        
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        if self.optim['SAM'] or self.optim['ASAM']:
            def closure():
                loss = self.loss_module(self(x), y)
                self.manual_backward(loss)
                return loss
            loss = self.loss_module(pred, y)
            self.manual_backward(loss, retain_graph=True)
            self.optimizers().step(closure=None)
            self.optimizers().zero_grad()
            self.log("train/loss", loss.item())
        else:
            loss = self.loss_module(pred, y)
            self.log("train/loss", loss.item())

        acc = self.accuracy(pred, y)
        self.log('train/Accuracy', acc, on_epoch=True, on_step=False, prog_bar=True)
        f1score = self.f1score(pred, y)
        self.log('train/F1Score', f1score, on_epoch=True, on_step=False, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)

        loss = self.loss_module(pred, y)
        self.log("val/loss", loss.item(), on_epoch=True, on_step=False)

        acc = self.accuracy(pred, y)
        self.log('val/Accuracy', acc, on_epoch=True, on_step=False, prog_bar=True)
        f1score = self.f1score(pred, y)
        self.log('val/F1Score', f1score, on_epoch=True, on_step=False, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch
        pred = self(x)
        self.test_preds.append(pred)

    def get_test_preds(self):
        return torch.concat(self.test_preds)

    def configure_optimizers(self):
        params = self.net.parameters()
        if self.lr_layer_decay != 1.0:
            params = self._get_llrd_params(lr=self.lr, layer_decay=self.lr_layer_decay, verbose=True)

        if self.optim['class_path']=='SGD':
            optimizer = optim.SGD(
                                params = params, 
                                lr = self.lr,
                                momentum = self.optim[self.optim['class_path']]['init_args']['momentum'],
                                weight_decay = self.optim[self.optim['class_path']]['init_args']['weight_decay'])
        if self.optim['class_path']=='AdamW':
            optimizer = optim.AdamW(
                                params = params,)
        if self.optim['SAM'] or self.optim['ASAM']:
            self.automatic_optimization = False
            optimizer = SAM(
                            params = params,
                            base_optimizer = SGD,
                            rho = self.lr,
                            adaptive = self.optim['ASAM'],
                            lr = self.lr,
                            momentum = self.optim[self.optim['class_path']]['init_args']['momentum'],
                            weight_decay = self.optim[self.optim['class_path']]['init_args']['weight_decay'])

        if self.lr_scheduler['class_path']=='CosineAnnealingLR':
            scheduler = lr_scheduler.CosineAnnealingLR(
                                optimizer, 
                                T_max=self.lr_scheduler['init_args']['T_max'], 
                                eta_min = 1e-6,
                                last_epoch=self.last_epoch)
        return [optimizer], [scheduler]

    def _get_llrd_params(self, lr, layer_decay=0.8, weight_decay=0.001, verbose=False):
        if verbose: print('\nSetting layer-wise learning rate...')
        n_encoder_layers = len(list(self.net.model.hubert.encoder.layers.named_children())) # this code can vary upon model's architecture. check model's architecture using model.named_parameters()
        llrd_params = []
        for name, value in list(self.named_parameters()):
            if 'encoder.layers' in name:
                for n_layer in range(n_encoder_layers):
                    if f'encoder.layers.{n_layer}.' in name:
                        llrd_params.append({"params": value, "lr": lr * (layer_decay**(n_encoder_layers - n_layer)), "weight_decay": weight_decay})
                        if verbose: print(name, lr * (layer_decay**(n_encoder_layers - n_layer)))
            elif ('emb' in name) or ('feature' in name): 
                llrd_params.append({"params": value, "lr": lr * (layer_decay**(n_encoder_layers+1)), "weight_decay": weight_decay})
                if verbose: print(name, lr* (layer_decay**(n_encoder_layers+1)))
            else:
                llrd_params.append({"params": value, "lr": lr , "weight_decay": weight_decay})
                if verbose: print(name, lr)
        return llrd_params