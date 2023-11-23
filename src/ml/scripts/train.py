import lightning as L
from ml.models.baseline import BaseLine
from ml.module.lightning_module import LitModule
from ml.data.datamodule import LitDataModule
from ml.utils.constants import CFG, EXPERIMENTS_DIR, DATA_DIR
from ml.transform.default import Transform

# transform = Transform(CFG['transform']['padding'])

model = LitModule(
                net = BaseLine(), 
                loss_module = CFG['model']['init_args']['loss_module']['class_path'],
                num_classes = CFG['model']['init_args']['net']['init_args']['num_classes'],
                optim_config = CFG['optimizer'],
                lr_schdlr_config = None,
                )

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

trainer = L.Trainer(
                logger = None,
                max_epochs = CFG['trainer']['n_epoch'],
                accelerator = CFG['trainer']['accelerator'],
                check_val_every_n_epoch = CFG['trainer']['check_val_every_n_epoch'],
                default_root_dir = EXPERIMENTS_DIR
                )

trainer.fit(model, 
            datamodule=datamodule,
            # ckpt_path=
            )