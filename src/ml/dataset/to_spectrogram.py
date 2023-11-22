import sys
from tqdm import tqdm
import torch
import torchaudio
import numpy as np
from glob import glob
from io import BytesIO

sys.path.append('/home/elicer/project')

from spectrogram import MelSpectrogram
from ml.utils.constants import CFG

CFG = CFG['mel_spectrogram']
raw_data_dir = '/home/elicer/project/data/raw/audio-mnist-whole'
extractor = MelSpectrogram(
                    sample_rate = CFG['sample_rate'], 
                    n_fft = CFG['n_fft'], 
                    win_length = CFG['win_length'], 
                    hop_length = CFG['hop_length'],
                    n_mel = CFG['n_mel_filter'],
                    pad = CFG['pad'], 
                    f_min = CFG['f_min'], 
                    f_max = CFG['f_max']
                    )

for file_name in tqdm(glob(raw_data_dir + '/*.wav')):
    x, sample_rate = torchaudio.load(file_name, format='wav', backend='soundfile')
    spectrogram = extractor(x)
    name = '/home/elicer/project/data/processed/audio-mnist-whole/MS/' + file_name.split('/')[-1].split('.')[0] + '.pt'
    torch.save(spectrogram.to('cpu'), name)