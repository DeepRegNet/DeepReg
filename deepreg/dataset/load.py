import os
from copy import deepcopy
from typing import Optional

from deepreg.dataset.loader.interface import DataLoader
from deepreg.registry import FILE_LOADER_CLASS, REGISTRY


def get_data_loader(data_config: dict, mode: str) -> Optional[DataLoader]:
    """
    Return the corresponding data loader.
    Can't be placed in the same file of loader interfaces as it causes import cycle.
    :param data_config: a dictionary containing configuration for data
    :param mode: string, must be train/valid/test
    :return: DataLoader or None, returns None if the data_dir_paths is empty
    """
    assert mode in ["train", "valid", "test"], "mode must be one of train/valid/test"

    data_dir_paths = data_config["dir"].get(mode, None)
    if data_dir_paths is None or data_dir_paths == "":
        return None
    if isinstance(data_dir_paths, str):
        data_dir_paths = [data_dir_paths]
    # replace ~ with user home path
    data_dir_paths = list(map(os.path.expanduser, data_dir_paths))
    for data_dir_path in data_dir_paths:
        if not os.path.isdir(data_dir_path):
            raise ValueError(
                f"Data directory path {data_dir_path} for mode {mode}"
                f" is not a directory or does not exist"
            )

    # prepare data loader config
    data_loader_config = deepcopy(data_config)
    data_loader_config.pop("dir")
    data_loader_config.pop("format")
    data_loader_config["name"] = data_loader_config.pop("type")

    default_args = dict(
        data_dir_paths=data_dir_paths,
        file_loader=REGISTRY.get(category=FILE_LOADER_CLASS, key=data_config["format"]),
        labeled=data_config["labeled"],
        sample_label="sample" if mode == "train" else "all",
        seed=None if mode == "train" else 0,
    )
    data_loader = REGISTRY.build_data_loader(
        config=data_loader_config, default_args=default_args
    )
    return data_loader
