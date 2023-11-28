import os, sys, argparse
import wandb
import pandas as pd
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner

sys.path.append('/home/elicer/project/src/ml')
from model import *
from module import LitModule
from data import LitDataModule
from transform import *
from util.constants import *

# -------- parsing argument --------
parser = argparse.ArgumentParser()
parser.add_argument('--train', dest='train', default=False, action='store_true')
parser.add_argument('--test', dest='test', default=False, action='store_true') 
parser.add_argument('--tune', dest='tune', default=False, action='store_true')
parser.add_argument('-n', '--name', dest='wandb_run_name', default='') # wandb run name. If None, do not use wandb logger
parser.add_argument('-c', '--ckptpath', dest='ckpt_path', default=None) # model checkpoint to resume training or to predct
parser.add_argument('--run-id', dest='wandb_run_id', default=None) # wandb run-id to resume training
parser.add_argument('--last-epoch', dest='last_epoch', default=None) # last epoch number in case of resuming a training
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

TRAIN_DATA_DIR = '/home/elicer/project/Emotions/' # must end with separator
TRAIN_CSV_PATH = '/home/elicer/project/Emotions/label.csv'
TEST_DATA_DIR = ''
TEST_CSV_PATH = ''

# TRAIN_DATA_DIR = '/home/elicer/project/월간 데이콘 음성 감정 인식 AI 경진대회/train/' # must end with separator
# TEST_DATA_DIR = '/home/elicer/project/월간 데이콘 음성 감정 인식 AI 경진대회/test/' # must end with separator
# TRAIN_CSV_PATH = '/home/elicer/project/월간 데이콘 음성 감정 인식 AI 경진대회/train.csv'
# TEST_CSV_PATH = '/home/elicer/project/월간 데이콘 음성 감정 인식 AI 경진대회/test.csv'

# TRAIN_DATA_DIR = '/home/elicer/project/data/raw/audio-mnist-whole/' # must end with separator
# TEST_DATA_DIR = '' # must end with separator
# TRAIN_CSV_PATH = '/home/elicer/project/data/raw/train.csv'
# TEST_CSV_PATH = ''

# -------------- main --------------
net_name = CFG['model']['class_path']
net = globals()[net_name](CFG['model']['num_classes'], **CFG['model'][net_name]['init_args'])

model = LitModule(
                net = net, 
                num_classes = CFG['model']['num_classes'],
                loss_module = CFG['model']['loss_module']['class_path'],
                lr = CFG['trainer']['lr'],
                lr_layer_decay = CFG['trainer']['lr_layer_decay'],
                optim = CFG['optimizer'],
                lr_scheduler = CFG['lr_scheduler'],
                )

transform_name = CFG['transform']['class_path']
transform = globals()[transform_name](args.train, **CFG['transform'][transform_name]['init_args'])

datamodule = LitDataModule(
                train_data_dir = TRAIN_DATA_DIR,
                train_csv_path = TRAIN_CSV_PATH,
                test_data_dir = TEST_DATA_DIR,
                test_csv_path = TEST_CSV_PATH,
                batch_size = CFG['data']['init_args']['batch_size'],
                val_split = CFG['data']['init_args']['val_split'],
                num_workers = CFG['data']['init_args']['num_worker'],
                transform = transform,
                collate_fn = transform.collate_fn if hasattr(transform, 'collate_fn') else None,
                on_memory = True
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
                        min_delta = 1e-5,
                        patience = 5,
                        verbose=True
                    ),
                    LearningRateMonitor(logging_interval = 'epoch'),
                ],
                num_sanity_val_steps = 1
                )


# -------------- task --------------
if args.tune:
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule = datamodule, max_lr=1e-2, min_lr=5e-7, early_stop_threshold=None)
    new_lr = lr_finder.suggestion()
    print("LR Suggestion => ", new_lr)
    fig = lr_finder.plot(suggest = True)
    fig.savefig('lr_finder.png')
    model.hparams.lr = new_lr

if args.train:
    torch.autograd.set_detect_anomaly(True)
    # wandb_logger.watch(model, log='all', log_freq=1)
    trainer.fit(model, 
                datamodule = datamodule, 
                ckpt_path = args.ckpt_path
                )

    wandb.finish()

if args.test:
    trainer.test(model,
                datamodule = datamodule,
                ckpt_path = args.ckpt_path)
    preds = model.get_test_preds().detach().cpu().numpy()
    preds =np.argmax(preds, axis=1)
    submission_df = pd.read_csv(TEST_CSV_PATH, header=0)
    submission_df['label'] = preds
    submission_df.to_csv('submission.csv', index=False)