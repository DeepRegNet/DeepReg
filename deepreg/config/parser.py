import collections.abc
import logging
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
    config_sanity_check(config)
    return config


def save(config: dict, out_dir: str, filename: str = "config.yaml"):
    assert filename.endswith(".yaml")
    with open(os.path.join(out_dir, filename), "w+") as f:
        f.write(yaml.dump(config))


def config_sanity_check(config: dict):
    """check if the given config satisfies the requirements"""

    # check data
    assert "data" in config.keys()
    data_config = config["data"]

    if data_config["type"] not in ["paired", "unpaired", "grouped"]:
        raise ValueError(
            "data type must be paired / unpaired / grouped,"
            "got {}.".format(data_config["type"])
        )

    if data_config["format"] not in ["nifti", "h5"]:
        raise ValueError(
            "data format must be nifti / h5," "got {}.".format(data_config["format"])
        )

    assert "dir" in data_config
    for mode in ["train", "valid", "test"]:
        assert mode in data_config["dir"].keys()
        data_dir = data_config["dir"][mode]
        if data_dir is None:
            logging.warning("Data directory for {} is not defined.".format(mode))
        if not (
            isinstance(data_dir, str) or isinstance(data_dir, list) or data_dir is None
        ):
            raise ValueError(
                "data_dir for mode {} must be string or list of strings,"
                "got {}.".format(mode, data_dir)
            )

    # check model
    if data_config["labeled"] is False:  # unlabeled
        image_loss_weight = config["tf"]["loss"]["dissimilarity"]["image"]["weight"]
        if image_loss_weight <= 0:
            raise ValueError(
                "For unlabeled data, the image loss must have positive weight, "
                "got {}.".format(image_loss_weight)
            )
