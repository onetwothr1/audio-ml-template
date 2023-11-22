import lightning as L
from ml.utils.constants import LOGGING_DIR

model = MyLightningModule()
datamodule = MyLightningDataModule()
trainer = L.Trainer(logger = TensorBoardLogger(EXPERIMENTS_DIR))
trainer.fit(model, data_module=datamodule)