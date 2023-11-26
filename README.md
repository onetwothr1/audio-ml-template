# audio-ml-template

Template code for audio model, using [Pytorch Lightning](https://github.com/Lightning-AI/lightning) and [wandb](https://github.com/wandb/wandb).

# Preparaton
First, excecute following line, in terminal. It ensures pyton to recognize its packages and modules.
```
export PYTHONPATH="${PYTHONPATH}:/path/of/the/project/src/directory"
```


# Prepare Data
* All the sound file data (`'*.wav'`) should be located in one directory, each for train and test. 
<br>
* CSV file with two columns, `'file_path'` and `'label'` should be prepared, each for train and test. `'file_path'` should be a path after the train(or test) directory. For example, a soundfile 'dataset/train/data1.wav's path on CSV is 'data1.wav'. `'label'` can be a string or integer.
<br>
* In `src/ml/script/train.py`, type the path of data directory and csv file, each for train and test.


# How To Train
1. To use `wandb` logger, Type your `wandb` api-key inside `config/wandb-api-key.txt`.
2. Modify `config/config.yaml` file as you want.
3. Modify `DATA_DIR` variable in `src/ml/util/constants.py` with the path of data for training. All the data should be inside that directory.
4. Run following line in a terminal.
```
python scripts/train.py -n 'version_name_for_a_run'
```

# EDA
Run following line in a terminal.
```
python utils/EDA.py
```

