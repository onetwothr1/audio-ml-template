model = MyLightningModule()
datamodule = MyLightningDataModule()
trainer = Trainer(logger = TensorBoardLogger(EXPERIMENTS_DIR))
trainer.test(model, data_module=datamodule)