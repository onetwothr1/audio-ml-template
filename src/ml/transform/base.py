import abc
import torchaudio

class Transform(abc.ABC):
    @abc.abstractmethod
    def train_transform(self, x):
        raise NotImplementedError
    
    @abc.abstractmethod
    def val_transform(self, x):
        raise NotImplementedError
    
    @abc.abstractmethod
    def test_transform(self, x):
        raise NotImplementedError

    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)