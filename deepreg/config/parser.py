import collections.abc
import logging
import os

import yaml

from deepreg.config.v011 import parse_v011


def update_nested_dict(d: dict, u: dict) -> dict:
    """
    Merge two dicts.

    https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth

    :param d: dict to be overwritten in case of conflicts.
    :param u: dict to be merged into d.
    :return:
    """

    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_configs(config_path: (str, list)) -> dict:
    """
    Load multiple configs and update the nested dictionary.

    :param config_path: list of paths or one path.
    :return:
    """
    if isinstance(config_path, str):
        config_path = [config_path]
    # replace ~ with user home path
    config_path = list(map(os.path.expanduser, config_path))
    config = dict()
    for config_path_i in config_path:
        with open(config_path_i) as file:
            config_i = yaml.load(file, Loader=yaml.FullLoader)
        config = update_nested_dict(d=config, u=config_i)
    loaded_config = config_sanity_check(config)

    if loaded_config != config:
        # config got updated
        head, tail = os.path.split(config_path[0])
        filename = "updated_" + tail
        save(config=loaded_config, out_dir=head, filename=filename)
        logging.error(
            f"Used config is outdated. "
            f"An updated version has been saved at "
            f"{os.path.join(head, filename)}"
        )

    return loaded_config


def save(config: dict, out_dir: str, filename: str = "config.yaml"):
    """
    Save the config into a yaml file.

    :param config: configuration to be outputed
    :param out_dir: directory of the output file
    :param filename: name of the output file
    """
    assert filename.endswith(".yaml")
    with open(os.path.join(out_dir, filename), "w+") as f:
        f.write(yaml.dump(config))


def config_sanity_check(config: dict) -> dict:
    """
    Check if the given config satisfies the requirements.

    :param config: entire config.
    """

    # check data
    data_config = config["dataset"]

    if data_config["type"] not in ["paired", "unpaired", "grouped"]:
        raise ValueError(f"data type must be paired / unpaired / grouped, got {type}.")

    if data_config["format"] not in ["nifti", "h5"]:
        raise ValueError(f"data format must be nifti / h5, got {format}.")

    assert "dir" in data_config
    for mode in ["train", "valid", "test"]:
        assert mode in data_config["dir"].keys()
        data_dir = data_config["dir"][mode]
        if data_dir is None:
            logging.warning(f"Data directory for {mode} is not defined.")
        if not (isinstance(data_dir, (str, list)) or data_dir is None):
            raise ValueError(
                f"data_dir for mode {mode} must be string or list of strings,"
                f"got {data_dir}."
            )

    # back compatibility support
    config = parse_v011(config)

    # check model
    if config["train"]["method"] == "conditional":
        if data_config["labeled"] is False:  # unlabeled
            raise ValueError(
                "For conditional model, data have to be labeled, got unlabeled data."
            )

    # loss weights should >= 0
    for name in ["image", "label", "regularization"]:
        loss_weight = config["train"]["loss"][name]["weight"]
        if loss_weight <= 0:
            logging.warning(
                "The %s loss weight %.2f is not positive.", name, loss_weight
            )

    return config
