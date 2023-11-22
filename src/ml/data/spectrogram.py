from torch import nn
import torchaudio.transforms as AT

class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate,
        n_fft,
        win_length,
        hop_length,
        pad,
        f_min,
        f_max,
        n_mel
    ) -> None:
        super().__init__()
        self.ms = nn.Sequential(
            AT.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                pad=pad,
                f_min=f_min,
                f_max=f_max,
                n_mels=n_mel
            ), 
            AT.AmplitudeToDB()
        )

    def forward(self, x):
        return self.ms(x)
        

# Multi-Resolution Mel-Spectrogram 
class MRMS(nn.Module):
    def __init__(self):
        pass

    def forard(self, x):
        return x