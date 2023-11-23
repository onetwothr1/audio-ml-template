import torch
import torch.nn.functional as F
import random
import numpy as np

def random_pad(mels: torch.tensor, fixed_size:int):
  pad_width = fixed_size - mels.shape[1]
  rand = np.random.rand()
  left = int(pad_width * rand)
  right = pad_width - left
  mels = torch.tensor(np.pad(mels, pad_width=((0,0), (left, right)), mode='constant'))
  return mels

def random_pad_batch(mels_batch: torch.tensor, fixed_size:int):
  pad_width = fixed_size - mels_batch.shape[2]
  rand = np.random.rand()
  left = int(pad_width * rand)
  right = pad_width - left
  mels_batch = torch.tensor(np.pad(mels_batch, pad_width=((0,0), (left, right)), mode='constant'), constant_values=0)
  return mels_batch

# 예제 데이터 생성 (1채널 이미지, batch_size=2)
data_batch = torch.rand(100, 20, 30)  # 크기가 (20, 30)인 1채널 이미지, batch_size=2

# 원하는 가로 크기로 조정 및 랜덤하게 패딩
fixed_size = 35  # 원하는 가로 크기
data_batch_padded = random_pad_batch(data_batch, fixed_size)
# 결과 확인
print("Original Size:", data_batch.size())
print("Fixed and Padded Size:", data_batch_padded.size())
