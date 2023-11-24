from torch.utils.data import Subset, DataLoader
import lightning as L

from .dataset import AudioDataset
from transform import Transform
from util.helpers import stratified_split


class LitDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_data_dir: str,
        test_data_dir: str,
        batch_size: int,
        val_split: float,
        num_workers: int,
        transform: Transform,
        collate_fn: callable=None
    ) -> None:
        super().__init__()
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.val_split = val_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = transform.train_transform
        self.val_transform = transform.val_transform
        self.test_transform = transform.test_transform
        self.collate_fn = collate_fn

    def setup(self, stage: str=None):
        dataset = AudioDataset(
            data_dir = self.train_data_dir,
            transform = self.train_transform)

        self.train_dataset, self.train_labels, self.test_dataset, self.test_labels = \
            stratified_split(dataset, dataset.make_labels(), val_split=0.1, random_state=42)
        self.train_dataset, self.train_labels, self.val_dataset, self.val_labels = \
            stratified_split(self.train_dataset, self.train_labels, val_split=self.val_split, random_state=42)

        self.train_dataset.transform = self.train_transform
        self.val_dataset.transform = self.val_transform
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
            self.test_dataset, batch_size=100, shuffle=False, num_workers=self.num_workers, collate_fn=self.collate_fn
        )

    # def prepare_data(self):
    #     # dowload data
    #     pass


