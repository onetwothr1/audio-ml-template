import os, sys, argparse
import wandb
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

sys.path.append('/home/elicer/project/src/ml')

from model import *
from module import LitModule
from data import LitDataModule
from transform import *
from util.constants import *

# -------- parsing argument --------
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--name', dest='wandb_run_name', default='') # wandb run name. If None, do not use wandb logger
parser.add_argument('-c', '--ckptpath', dest='ckpt_path', default=None) # model checkpoint to resume training
parser.add_argument('--run-id', dest='wandb_run_id', default=None) # wandb run-id to resume training

args = parser.parse_args()


# ----------- preparation ----------
seed_everything(CFG['seed_everything'], workers=True)

if args.wandb_run_name:
    wandb.login(key = WANDB_API_KEY)
    wandb_logger = WandbLogger(
                    project = CFG['project'],
                    name = args.wandb_run_name,
                    id = args.wandb_run_id,
                    config = CFG,
                    )


# -------------- main --------------
net_name = CFG['model']['class_path']
net = globals()[net_name](CFG['model']['num_classes'], **CFG['model'][net_name]['init_args'])

model = LitModule(
                net = net, 
                num_classes = CFG['model']['num_classes'],
                loss_module = CFG['model']['loss_module']['class_path'],
                lr = CFG['trainer']['lr'],
                optim = CFG['optimizer'],
                lr_scheduler = CFG['lr_scheduler'],
                )

transform_name = CFG['transform']['class_path']
transform = globals()[transform_name](**CFG['transform'][transform_name]['init_args'])

datamodule = LitDataModule(
                train_data_dir = DATA_DIR,
                test_data_dir = None,
                batch_size = CFG['data']['init_args']['batch_size'],
                val_split = CFG['data']['init_args']['val_split'],
                num_workers = CFG['data']['init_args']['num_worker'],
                transform = transform,
                collate_fn = transform.collate_fn if hasattr(transform, 'collate_fn') else None
                )

trainer = L.Trainer(
                logger = wandb_logger if args.wandb_run_name else None,
                max_epochs = CFG['trainer']['n_epoch'],
                accelerator = CFG['trainer']['accelerator'],
                check_val_every_n_epoch = CFG['trainer']['check_val_every_n_epoch'],
                precision = CFG['trainer']['precision'],
                deterministic = False, # MUST BE FALSE WHEN ITS WAV2VEC2. FATAL ERROR RAISES
                callbacks = [
                    ModelCheckpoint(
                        save_top_k = 5,
                        monitor = 'val/loss',
                        mode = 'min',
                        dirpath = EXPERIMENTS_DIR + os.sep + CFG['project'] + os.sep + args.wandb_run_name,
                        filename = "{epoch:02d}-{val_loss:.4f}"
                    ),
                    EarlyStopping(
                        monitor = 'val/loss',
                        mode = 'min',
                        min_delta = 1e-4,
                        patience = 4,
                    ),
                    LearningRateMonitor(logging_interval = 'epoch'),
                    DeviceStatsMonitor()
                ],
                num_sanity_val_steps = 1
                )

trainer.fit(model, 
            datamodule = datamodule, 
            ckpt_path = args.ckpt_path
            )

wandb.finish()