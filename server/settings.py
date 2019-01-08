import pathlib
import json

BASE_DIR = pathlib.Path(__file__).parent
config_path = BASE_DIR / 'config.json'


def get_config(path):
    with open(path) as f:
        config = json.load(f)
    return config

config = get_config(config_path)