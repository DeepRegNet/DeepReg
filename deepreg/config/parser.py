import os

import yaml


def load(config_path):
    with open(config_path) as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def save(config: dict, out_dir: str, filename: str = "config.yaml"):
    assert filename.endswith(".yaml")
    with open(os.path.join(out_dir, filename), "w+") as f:
        f.write(yaml.dump(config))
