
import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm


def audio_length_distibution(data_dir, ext='wav'):
    ms_list = []
    for file_name in tqdm(glob(data_dir + '/*.' + ext)):
        sig, sr = torchaudio.load(file_name)
        ms_list.append(sig.size(1)/sr)
    ms_list.sort()
    plt.hist(ms_list, bins=200)

audio_length_distibution('/home/elicer/project/data/raw/audio-mnist-whole')