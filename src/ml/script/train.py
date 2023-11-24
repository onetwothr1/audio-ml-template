import os, sys, argparse
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

sys.path.append('/home/elicer/project/src/ml')

from model import BaseLine
from module import LitModule
from data import LitDataModule
from transform import Transform
from util.constants import *

# ---------------------
# parsing argument
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name', default=None, required=True)
# to resume training from a checkpoint
parser.add_argument('-c', '--ckptpath', dest='ckpt_path', default=None) 
parser.add_argument('--run-id', dest='wandb_run_id', default=None)
args = parser.parse_args()

# ---------------------
# model
# ---------------------
model = LitModule(
                net = BaseLine(), 
                loss_module = CFG['model']['init_args']['loss_module']['class_path'],
                num_classes = CFG['model']['init_args']['net']['init_args']['num_classes'],
                optim = CFG['optimizer'],
                lr_scheduler = CFG['lr_scheduler'],
                )


datamodule = LitDataModule(
                train_data_dir=DATA_DIR,
                test_data_dir=None,
                batch_size = CFG['data']['init_args']['batch_size'],
                val_split = CFG['data']['init_args']['val_split'],
                num_workers = CFG['data']['init_args']['num_worker'],
                transform = Transform(
                    audio_max_ms = CFG['transform']['audio_max_ms'],
                    sample_rate = CFG['transform']['sample_rate'],
                    mel_spectrogram = CFG['transform']['mel_spectrogram'],
                    time_shift = CFG['transform']['time_shift'],
                    masking = CFG['transform']['masking'],
                    noising = CFG['transform']['noising']
                )
                )


wandb.login(key = WANDB_API_KEY)
wandb_logger = WandbLogger(
                project = CFG['project'],
                name = args.name,
                id = args.wandb_run_id,
                config = CFG,
                )

trainer = L.Trainer(
                logger = wandb_logger,
                max_epochs = CFG['trainer']['n_epoch'],
                accelerator = CFG['trainer']['accelerator'],
                check_val_every_n_epoch = CFG['trainer']['check_val_every_n_epoch'],
                callbacks = [
                    ModelCheckpoint(
                        save_top_k = 5,
                        monitor = 'val/loss',
                        mode = 'min',
                        dirpath = EXPERIMENTS_DIR + os.sep + CFG['project'] + os.sep + args.name,
                        filename = "{epoch:02d}-{val_loss:.4f}"
                    ),
                    EarlyStopping(
                        monitor = 'val/loss',
                        mode = 'min',
                        min_delta = 1e-4,
                        patience = 4,
                    ),
                    LearningRateMonitor(logging_interval = 'epoch')
                ]
                )

trainer.fit(model, 
            datamodule = datamodule, 
            ckpt_path = args.ckpt_path
            )
wandb.finish()