import sys
from tqdm import tqdm
import torchaudio
from spectrogram import MelSpectrogram
from ml.utils.constants import DEVICE, CFG

CFG = CFG['mel_spectrogram']
data_dir = '/home/elicer/project/data/raw/audio-mnist-whole'
extractor = MelSpectrogram(
                    sample_rate = CFG['sample_rate'], 
                    n_fft = CFG['n_fft'], 
                    win_length = CFG['win_length'], 
                    hop_length = CFG['hop_length'],
                    n_mel = CFG['n_mel'],
                    pad = CFG['pad'], 
                    f_min = CFG['f_min'], 
                    f_max = CFG['f_max']
                    )

for file_name in tqdm(np.sort(glob(data_dir + '/*.wav'))):
    x,_ = torchaudio.load(file_name)
    x = x.to(DEVICE)
    spectrogram = extractor(x)
    name = '/home/elicer/project/data/processed/audio-mnist-whole' + file_name.slit('/')[-1].split('.')[0] + '.pt'
    torch.save(spectrogram.to('cpu'), name)
    break