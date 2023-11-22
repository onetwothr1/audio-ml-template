import lightning as L
from ml.models.baseline import BaseLine
from ml.module.lightning_module import LitModule
from ml.data.datamodule import LitDataModule
from ml.utils.constants import LOGGING_DIR, CFG

model = LitModule(BaseLine(CFG['model']), **CFG['model'])
datamodule = LitDataModule(**CFG['data'])
trainer = L.Trainer(logger = None)
trainer.fit(model, data_module=datamodule)