import abc

class BaseTransforms(abc.ABC):
    @abc.abstractmethod
    def train_transform(self, x):
        raise NotImplementedError
    
    @abc.abstractmethod
    def val_transform(self, x):
        raise NotImplementedError
    
    @abc.abstractmethod
    def test_transform(self, x):
        raise NotImplementedError