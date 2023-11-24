import yaml, os
DATA_DIR = "/home/elicer/project/data/raw/audio-mnist-whole"
with open('/home/elicer/project/config/config.yaml') as f:
    CFG = yaml.load(f, Loader=yaml.FullLoader)

print(CFG['transform']['noising']['use'])
print(type(CFG['transform']['noising']['use']))