#%%
import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
from matplotlib import pyplot as plt
from glob import glob
from tqdm import tqdm
import wave

data_dir = '/home/elicer/project/data/raw/audio-mnist-whole'
ext = 'wav'

def EDA():
    ms_list = [] # audio length in milisec
    sr_list = [] # sample rate
    ch_list = [] # channel num
    for file_name in tqdm(glob(data_dir + '/*.' + ext)):
        sig, sr = torchaudio.load(file_name)
        ms_list.append(sig.size(1)/sr)
        sr_list.append(sr)
        ch_list.append(sig.size(0))
    plt.hist(ms_list, bins=200)
    plt.xlabel('ms')
    plt.ylabel('num data')
    plt.title('audio lenth distribution')
    plt.show()

    sr_all_same = all(sr == sr_list[0] for sr in sr_list)
    if sr_all_same:
        print("SAMPLE RATE: ", sr_list[0])
    else:
        print("SAMPLE RATEs are not equal")
        plt.hist(sr_list)
        plt.xlabel('sample rate')
        plt.ylabel('num data')
        plt.title('samplerate')
        plt.show()


    ch_all_same = all(ch == ch_list[0] for ch in ch_list)
    if ch_all_same:
        print("CHANNEL: ", ch_list[0])
    else:
        mono = 0
        stereo = 0
        for i in ch_list:
            if i==1:
                mono += 1
            elif i==2:
                stereo += 1
            else:
                print('channel should be 1 or 2 but got channel size of ', i)
                break
        print("CHANNELs are not equal")
        print('mono: ', mono)
        print('stereo', stereo)

if __name__=='__main__':
    EDA()
# %%
