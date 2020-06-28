from deepreg.dataset.loader.grouped_loader import GroupedDataLoader
from deepreg.dataset.loader.h5_loader import H5FileLoader
from deepreg.dataset.loader.interface import ConcatenatedDataLoader
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader
from deepreg.dataset.loader.paired_loader import PairedDataLoader
from deepreg.dataset.loader.unpaired_loader import UnpairedDataLoader

FileLoaderDict = dict(nifti=NiftiFileLoader, h5=H5FileLoader)


def get_data_loader(data_config, mode):
    """
    Return the corresponding data loader.
    Can't be placed in the same file of loader interfaces as it causes import cycle.
    :param data_config:
    :param mode:
    :return:
    """

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

    data_loaders = []
    for data_dir_path in data_dir_paths:
        data_loader_i = get_single_data_loader(
            data_type, data_config, common_args, data_dir_path
        )
        data_loaders.append(data_loader_i)

    if len(data_loaders) == 1:
        return data_loaders[0]
    else:
        return ConcatenatedDataLoader(data_loaders=data_loaders)


def get_single_data_loader(data_type, data_config, common_args, data_dir_path):
    if data_type == "paired":
        moving_image_shape = data_config["moving_image_shape"]
        fixed_image_shape = data_config["fixed_image_shape"]
        return PairedDataLoader(
            data_dir_path=data_dir_path,
            moving_image_shape=moving_image_shape,
            fixed_image_shape=fixed_image_shape,
            **common_args,
        )
    elif data_type == "grouped":
        image_shape = data_config["image_shape"]
        intra_group_prob = data_config["intra_group_prob"]
        intra_group_option = data_config["intra_group_option"]
        sample_image_in_group = data_config["sample_image_in_group"]
        return GroupedDataLoader(
            data_dir_path=data_dir_path,
            intra_group_prob=intra_group_prob,
            intra_group_option=intra_group_option,
            sample_image_in_group=sample_image_in_group,
            image_shape=image_shape,
            **common_args,
        )
    elif data_type == "unpaired":
        image_shape = data_config["image_shape"]
        return UnpairedDataLoader(
            data_dir_path=data_dir_path, image_shape=image_shape, **common_args
        )
    else:
        raise ValueError(
            "Unknown data format. "
            "Supported types are paired, unpaired, and grouped, got {}\n".format(
                data_type
            )
        )
