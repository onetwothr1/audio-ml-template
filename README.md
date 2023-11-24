# audio-ml-template

Template code for audio model, using [Pytorch Lightning](https://github.com/Lightning-AI/lightning) and [wandb](https://github.com/wandb/wandb).

# Preparaton
First, excecute following line, in terminal. It ensures pyton to recognize its packages and modules.
```
export PYTHONPATH="${PYTHONPATH}:/path/of/the/project/src/directory"
```


# To Train
1. Type your `wandb` api-key inside `config/wandb-api-key.txt`.
2. Modify `config/config.yaml` file as you want.
```
python scripts/train.py -n 'version_name_for_a_run'
```
