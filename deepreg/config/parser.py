import collections.abc
import os

import yaml


def update_nested_dict(d, u):
    """https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth"""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_configs(config_path: (str, list)):
    """load multiple configs and update the nested dictionary"""
    if isinstance(config_path, str):
        config_path = [config_path]
    config = dict()
    for config_path_i in config_path:
        with open(config_path_i) as file:
            config_i = yaml.load(file, Loader=yaml.FullLoader)
        config = update_nested_dict(d=config, u=config_i)
    return config


def save(config: dict, out_dir: str, filename: str = "config.yaml"):
    assert filename.endswith(".yaml")
    with open(os.path.join(out_dir, filename), "w+") as f:
        f.write(yaml.dump(config))
