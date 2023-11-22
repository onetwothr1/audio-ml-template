from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader

import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

class LitDataModule(L.MyLightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        val_split: Union[int, float],
        # train_transform=None,
        # val_transform=None,
        # test_transform=None,
        transforms: BaseTransforms,
    ):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.train_transform = transforms.train_transform()
        self.val_transform = transforms.val_transform()
        self.test_transform = transforms.test_transform()
        self.val_split = val_split
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.train_dataset = 
        self.val_dataset = 
        self.test_dataset = 

        indices = list(range(len(self.train_dataset)))
        targets = list(self.train_dataset.targets)

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.val_split, random_state=0
        )

        train_indices, val_indices = next(sss.split(indices, targets))

        self.train_dataset = Subset(self.train_dataset, train_indices)
        self.val_dataset = Subset(self.val_dataset, val_indices)

    def prepare_data(self):
        # dowload data
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=8
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=100, shuffle=False, num_workers=8
        )