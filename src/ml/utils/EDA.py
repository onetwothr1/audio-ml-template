import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm
import wave


def audio_length_distibution(data_dir, ext='wav'):
    ms_list = []
    for file_name in tqdm(glob(data_dir + '/*.' + ext)):
        sig, sr = torchaudio.load(file_name)
        ms_list.append(sig.size(1)/sr)
    ms_list.sort()
    plt.hist(ms_list, bins=200)
    plt.show()

def check_samplerate(data_dir, ext='wav'):
    sr_list = []
    for file_name in tqdm(glob(data_dir + '/*.' + ext)):
        with wave.open(data_dir + '/' + file_name, "rb") as wave_file:
            sr_list.append(wave_file.getframerate())

    all_same = all(sr == sr_list[0] for sr in sr_list)
    if all_same:
        print("SAMPLE RATE: ", sr_list[0])
    else:
        print("SAMPLE RATEs are not equal")
        plt.hist(sr_list)
        plt.xlabel('sample rate')
        plt.ylabel('num data')
        plt.show()
            
audio_length_distibution('/home/elicer/project/data/raw/audio-mnist-whole')