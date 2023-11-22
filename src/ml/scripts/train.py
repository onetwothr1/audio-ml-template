import lightning as L
from ml.models.baseline import BaseLine

from ml.utils.constants import LOGGING_DIR, CFG
model = BaseLine(CFG['model'])
datamodule = MyLightningDataModule(CFG['data'])
trainer = L.Trainer(logger = None)
trainer.fit(model, data_module=datamodule)