import os
from copy import deepcopy
from typing import Optional

from deepreg.constant import KNOWN_DATA_SPLITS
from deepreg.dataset.loader.interface import DataLoader
from deepreg.registry import FILE_LOADER_CLASS, REGISTRY


def get_data_loader(data_config: dict, split: str) -> Optional[DataLoader]:
    """
    Return the corresponding data loader.

    Can't be placed in the same file of loader interfaces as it causes import cycle.

    :param data_config: a dictionary containing configuration for data
    :param split: must be train/valid/test
    :return: DataLoader or None, returns None if the split or dir is empty.
    """
    if split not in KNOWN_DATA_SPLITS:
        raise ValueError(f"split must be one of {KNOWN_DATA_SPLITS}, got {split}")

    if split not in data_config:
        return None
    data_dir_paths = data_config[split].get("dir", None)
    if data_dir_paths is None or data_dir_paths == "":
        return None

    if isinstance(data_dir_paths, str):
        data_dir_paths = [data_dir_paths]
    # replace ~ with user home path
    data_dir_paths = list(map(os.path.expanduser, data_dir_paths))
    for data_dir_path in data_dir_paths:
        if not os.path.isdir(data_dir_path):
            raise ValueError(
                f"Data directory path {data_dir_path} for split {split}"
                f" is not a directory or does not exist"
            )

    # prepare data loader config
    data_loader_config = deepcopy(data_config)
    data_loader_config = {
        k: v for k, v in data_loader_config.items() if k not in KNOWN_DATA_SPLITS
    }
    data_loader_config["name"] = data_loader_config.pop("type")

    default_args = dict(
        data_dir_paths=data_dir_paths,
        file_loader=REGISTRY.get(
            category=FILE_LOADER_CLASS, key=data_config[split]["format"]
        ),
        labeled=data_config[split]["labeled"],
        sample_label="sample" if split == "train" else "all",
        seed=None if split == "train" else 0,
    )
    data_loader: DataLoader = REGISTRY.build_data_loader(
        config=data_loader_config, default_args=default_args
    )
    return data_loader
