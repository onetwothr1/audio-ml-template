import random
import math
import torch.utils.data
from collections import defaultdict

from util.constants import *

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

def update_config_yaml(name, new_value):
    CFG[name] = new_value
    with open(os.path.join(CONFIG_DIR, 'config.yaml'), 'w') as f:
        yaml.dump(CFG, f)