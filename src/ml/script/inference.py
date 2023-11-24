model = MyLightningModule()
datamodule = MyLightningDataModule()
trainer = Trainer(logger = TensorBoardLogger(EXPERIMENTS_DIR))
trainer.predict(model, data_module=datamodule)