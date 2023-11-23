from ml.transform.base import BaseTransforms

class Transform(BaseTransforms):
    def __init__(
        self,
        padding_config
    ) -> None:
        super().__init__()
        self.padding_config = padding_config

    def train_transform(self, x):
        print(x.size())
        x = random_pad(x, self.padding_config['fixed_size'])
        print(x.size())
        return x

    def val_transform(self, x):
        x = random_pad(x, self.padding_config['fixed_size'])
        return x
    
    def test_transform(self, x):
        x = random_pad(x, self.padding_config['fixed_size'])
        return x