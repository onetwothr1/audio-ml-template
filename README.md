# audio-ml-template

Template code for audio model, using [Pytorch Lightning](https://github.com/Lightning-AI/lightning) and [wandb](https://github.com/wandb/wandb).

# Preparaton
First, excecute following line, in terminal. It ensures pyton to recognize its packages and modules.
```
export PYTHONPATH="${PYTHONPATH}:/path/of/the/project/src/directory"
```


# Prepare Data
* All the sound file data (`'*.wav'`) should be located in one directory, each for train and test. 
* CSV file with two columns, `'file_path'` and `'label'` should be prepared, each for train and test. `'file_path'` should be a path after the train(or test) directory. For example, a soundfile 'dataset/train/data1.wav's path on CSV is 'data1.wav'. `'label'` should be an integer.
* In `src/ml/script/main.py`, type the path of data directory and csv file, each for train and test.


# Train
1. To use `wandb` logger, Type your `wandb` api-key inside `config/wandb-api-key.txt`.
2. Modify `config/config.yaml` file as you want.
3. Modify `DATA_DIR` variable in `src/ml/util/constants.py` with the path of data for training. All the data should be inside that directory.
4. Run following line in a terminal.
```
python scripts/main.py --train
```

## Wandb logger 
```
python scripts/main.py --train -n 'wandb-run-name'
```

## Resume training
```
python scripts/main.py --train -c 'model checkpoint file path' --run-id 'run-id of previous wandb run --last-epoch 'last epoch number'
```
Can omit '--run-id' if not using wandb. 
<br>Can omit '--last-epoch' if not using learning rate scheduler.
<br>
## LR_finder
```
python scripts/main.py --train --tune
```
It will automatically find the best lr and start training with it. Graph image of lr_finder is saved in `lr_finder.png`.

# Inference
```
python scripts/main.py --test -c 'model checkpoint file path'
```
It saves `submission.csv` file in a root directory.

# EDA
```
python utils/EDA.py
```


# Note
If you encounter an error while using a Hugging Face model, in most cases, reducing the batch size resolves the issue.
