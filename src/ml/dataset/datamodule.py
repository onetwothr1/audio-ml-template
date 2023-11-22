from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
import lightning as L
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
import random
import math
import torch.utils.data
from collections import defaultdict

from dataset import CustomDataset
from ml.utils.constants import CFG

CFG = CFG['data']
data_dir = '/home/elicer/project/data/processed/audio-mnist-whole/MS'

class LitDataModule(L.MyLightningDataModule):
    def __init__(
        self,
        root: str,
        batch_size: int,
        val_split: float,
    ) -> None:
        super().__init__()
        self.root = root
        self.val_split = val_split
        self.batch_size = batch_size

    def setup(self, stage: str=None):
        dataset = CustomDataset(train_data_dir)
        self.train_dataset, self.test_dataset = stratified_split(dataset, labels, fraction=0.1, random_state=42)
        self.train_dataset, self.val_dataset = stratified_split(self.train_datasetdataset, labels, fraction=self.val_split, random_state=42)

    # def prepare_data(self):
    #     # dowload data
    #     pass

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


def stratified_split(dataset : torch.utils.data.Dataset, labels, val_split, random_state=42):
    '''
    https://gist.github.com/Alvtron/9b9c2f870df6a54fda24dbd1affdc254
    '''
    fraction = 1 - val_split
    if random_state: random.seed(random_state)
    indices_per_label = defaultdict(list)
    for index, label in enumerate(labels):
        indices_per_label[label].append(index)
    first_set_indices, second_set_indices = list(), list()
    for label, indices in indices_per_label.items():
        n_samples_for_label = round(len(indices) * fraction)
        random_indices_sample = random.sample(indices, n_samples_for_label)
        first_set_indices.extend(random_indices_sample)
        second_set_indices.extend(set(indices) - set(random_indices_sample))
    first_set_inputs = torch.utils.data.Subset(dataset, first_set_indices)
    first_set_labels = list(map(labels.__getitem__, first_set_indices))
    second_set_inputs = torch.utils.data.Subset(dataset, second_set_indices)
    second_set_labels = list(map(labels.__getitem__, second_set_indices))
    return first_set_inputs, first_set_labels, second_set_inputs, second_set_labels