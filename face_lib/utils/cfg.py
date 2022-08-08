import yaml
from easydict import EasyDict


def load_config(path):
    with open(path) as fin:
        config = EasyDict(yaml.safe_load(fin))
    return config
