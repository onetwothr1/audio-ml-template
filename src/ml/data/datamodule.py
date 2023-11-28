import pandas as pd
import lightning as L
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from .dataset import AudioDataset
from transform import Transform

class LitDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        train_csv_path: str,
        test_data_dir: str,
        test_csv_path: str,
        transform: Transform,
        batch_size: int =32,
        val_split: float =0.2,
        num_workers: int =1,
        on_memory: bool=False,
        collate_fn: callable=None
    ) -> None:
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.train_df = pd.read_csv(train_csv_path, header=0)
        self.test_df = pd.read_csv(test_csv_path, header=0) if test_csv_path else None

        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = transform.train_transform
        self.val_transform = transform.val_transform
        self.test_transform = transform.test_transform
        self.on_memory = on_memory
        self.collate_fn = collate_fn

    def setup(self, stage: str=None):
        if stage=='fit' or stage is None:
            self.train_dataset = AudioDataset(
                data_dir = self.train_data_dir,
                df = self.train_df,
                transform = self.train_transform,
                on_memory=self.on_memory)
        
            sss = StratifiedShuffleSplit(n_splits=1, test_size=self.val_split)
            indices = list(range(len(self.train_df)))
            train_labels = self.train_df['label']
            train_indices, val_indices = next(sss.split(indices, train_labels))

            self.train_dataset = AudioDataset(
                data_dir = self.train_data_dir,
                df = self.train_df.loc[train_indices],
                transform = self.train_transform,
            )
            self.val_dataset = AudioDataset(
                data_dir = self.train_data_dir,
                df = self.train_df.loc[val_indices],
                transform = self.val_transform,
            )

            self.train_dataset.transform = self.train_transform
            self.val_dataset.transform = self.val_transform

        if stage=='test' or stage is None:
            self.test_dataset = AudioDataset(
                data_dir = self.test_data_dir,
                df = self.test_df,
                transform = self.test_transform,
                test = True)
            self.test_dataset.transform = self.test_transform

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    # def prepare_data(self):
    #     # dowload data
    #     pass