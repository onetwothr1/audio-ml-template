import os, sys, argparse
import wandb
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from ml.models.baseline import BaseLine
from ml.module.lightning_module import LitModule
from ml.data.datamodule import LitDataModule
from ml.utils.constants import CFG, EXPERIMENTS_DIR, DATA_DIR, WANDB_API_KEY
from ml.utils.helpers import update_config_yaml
from ml.transform.default import Transform

# ---------------------
# parsing argument
# ---------------------
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='name', default=None, action='store')
parser.add_argument('-c', '--ckptpath', dest='ckpt_path', default=None, action='store') # to resume training from a checkpoint
args = parser.parse_args()
# if args.name:
#     update_config_yaml('name', args.name)


# ---------------------
# model
# ---------------------
model = LitModule(
                net = BaseLine(), 
                loss_module = CFG['model']['init_args']['loss_module']['class_path'],
                num_classes = CFG['model']['init_args']['net']['init_args']['num_classes'],
                optim = CFG['optimizer'],
                lr_schduler = CFG['lr_scheduler'],
                )

# transform = Transform(CFG['transform']['padding'])

datamodule = LitDataModule(
                train_data_dir=DATA_DIR,
                test_data_dir=None,
                batch_size = CFG['data']['init_args']['batch_size'],
                val_split = CFG['data']['init_args']['val_split'],
                num_workers = CFG['data']['init_args']['num_worker'],
                audio_max_ms = CFG['data']['dataset']['init_args']['audio_max_ms'],
                mel_spectrogram = CFG['data']['mel_spectrogram'],
                # transform = transform
                )


wandb.login(key = WANDB_API_KEY)
wandb_logger = WandbLogger(
                project = CFG['project'],
                name = args.name,
                # name = CFG['name'],
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
                        dirpath = EXPERIMENTS_DIR + os.sep + CFG['project'] + os.sep + CFG['name'],
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