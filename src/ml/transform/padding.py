import torch
import torch.nn.functional as F
import random
import numpy as np

def random_pad_per_data(mels: torch.tensor, fixed_size:int):
  # mels: [height, width]
  pad_width = fixed_size - mels.shape[1]
  rand = np.random.rand()
  left = int(pad_width * rand)
  right = pad_width - left
  mels = torch.tensor(np.pad(mels, pad_width=((0,0), (left, right)), mode='constant'))
  return mels

def random_pad(mels_batch: torch.tensor, fixed_size:int):
  # mels: [batch_size, height, width]
  pad_size = fixed_size - mels_batch.shape[2]
  rand = np.random.rand()
  left = int(pad_size * rand)
  right = pad_size - left
  pad_width = [(0,0)] * (mels_batch.ndim - 1) + [(left, right)]
  mels_batch = torch.tensor(np.pad(mels_batch, pad_width=pad_width, mode='constant', constant_values=0))
  return mels_batch