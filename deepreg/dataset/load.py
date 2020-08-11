import os
from typing import List

from deepreg.dataset.loader.grouped_loader import GroupedDataLoader
from deepreg.dataset.loader.h5_loader import H5FileLoader
from deepreg.dataset.loader.interface import DataLoader
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader
from deepreg.dataset.loader.paired_loader import PairedDataLoader
from deepreg.dataset.loader.unpaired_loader import UnpairedDataLoader

FileLoaderDict = dict(nifti=NiftiFileLoader, h5=H5FileLoader)


def get_data_loader(data_config: dict, mode: str) -> (DataLoader, None):
    """
    Return the corresponding data loader.
    Can't be placed in the same file of loader interfaces as it causes import cycle.
    :param data_config: a dictionary containing configuration for data
    :param mode: string, must be "train"/"valid"/"test"
    :return: DataLoader or None, returns None if the data_dir_paths is empty
    """
    assert mode in ["train", "valid", "test"], "mode must be one of train/valid/test"
    data_type = data_config["type"]
    common_args = dict(
        file_loader=FileLoaderDict[data_config["format"]],
        labeled=data_config["labeled"],
        sample_label="sample" if mode == "train" else "all",
        seed=None if mode == "train" else 0,
    )

    data_dir_paths = data_config["dir"][mode]
    if data_dir_paths is None or data_dir_paths == "":
        return None
    if isinstance(data_dir_paths, str):
        data_dir_paths = [data_dir_paths]
    for data_dir_path in data_dir_paths:
        if not os.path.isdir(data_dir_path):
            raise ValueError(
                f"Data directory path {data_dir_path} for mode {mode} is not a directory or does not exist"
            )

    return get_single_data_loader(
        data_type=data_type,
        data_config=data_config,
        common_args=common_args,
        data_dir_paths=data_dir_paths,
    )


def get_single_data_loader(
    data_type: str, data_config: dict, common_args: dict, data_dir_paths: List[str]
) -> DataLoader:
    """
    Return one single data loader.
    :param data_type: type of the data, paired / unpaired / grouped
    :param data_config: dictionary containing the configuration of the data
    :param common_args: some shared arguments for all data loaders
    :param data_dir_paths: paths of the directories containing data
    :return: a basic data loader
    """
    assert isinstance(
        data_dir_paths, list
    ), f"data_dir_paths must be list of strings, got {data_dir_paths}"
    try:
        if data_type == "paired":
            moving_image_shape = data_config["moving_image_shape"]
            fixed_image_shape = data_config["fixed_image_shape"]
            return PairedDataLoader(
                data_dir_paths=data_dir_paths,
                moving_image_shape=moving_image_shape,
                fixed_image_shape=fixed_image_shape,
                **common_args,
            )
        elif data_type == "unpaired":
            image_shape = data_config["image_shape"]
            return UnpairedDataLoader(
                data_dir_paths=data_dir_paths, image_shape=image_shape, **common_args
            )
        elif data_type == "grouped":
            image_shape = data_config["image_shape"]
            intra_group_prob = data_config["intra_group_prob"]
            intra_group_option = data_config["intra_group_option"]
            sample_image_in_group = data_config["sample_image_in_group"]
            return GroupedDataLoader(
                data_dir_paths=data_dir_paths,
                intra_group_prob=intra_group_prob,
                intra_group_option=intra_group_option,
                sample_image_in_group=sample_image_in_group,
                image_shape=image_shape,
                **common_args,
            )
    except KeyError as e:
        msg = f"{e.args[0]} is not provided in the dataset config for paired data.\n"
        if data_type == "paired":
            msg += (
                "Paired Loader requires 'moving_image_shape' and 'fixed_image_shape'.\n"
            )
        elif data_type == "unpaired":
            msg += (
                "Unpaired Loader requires 'image_shape', "
                "as the data are not paired and will be resized to the same shape.\n"
            )
        elif data_type == "grouped":
            msg += (
                "Grouped Loader requires 'image_shape', "
                "as the data are not paired and will be resized to the same shape.\n"
                "It also requires 'intra_group_prob', 'intra_group_option', and 'sample_image_in_group'.\n"
            )
        raise ValueError(f"{msg}" f"The given dataset config is {data_config}\n")
    raise ValueError(
        f"Unknown data format {data_type}. "
        f"Supported types are paired, unpaired, and grouped.\n"
    )
