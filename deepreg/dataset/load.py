import os

from deepreg.dataset.loader.grouped_loader import GroupedDataLoader
from deepreg.dataset.loader.h5_loader import H5FileLoader
from deepreg.dataset.loader.nifti_loader import NiftiFileLoader
from deepreg.dataset.loader.paired_loader import PairedDataLoader
from deepreg.dataset.loader.unpaired_loader import UnpairedDataLoader


def get_data_loader(data_config, mode):
    """
    Return the corresponding data loader.
    Can't be placed in the same file of loader interfaces as it causes import cycle.
    :param data_config:
    :param mode:
    :return:
    """
    data_dir = data_config["dir"]

    # set mode
    modes = os.listdir(data_dir)
    if "train" not in modes:
        raise ValueError(
            "training data must be provided, they should be stored under train/"
        )
    if mode == "valid" and mode not in modes:
        # when validation data is not available, use test data instead
        mode = "test"
    if mode not in modes:
        raise ValueError("Unknown mode {}. Supported modes are {}".format(mode, modes))

    data_type = data_config["type"]

    # sanity check for configs
    # TODO move checks
    if data_type not in ["paired", "unpaired", "grouped"]:
        raise ValueError("data type must be paired / unpaired / grouped")

    if data_config["format"] == "nifti":
        file_loader = NiftiFileLoader
    elif data_config["format"] == "h5":
        file_loader = H5FileLoader
    else:
        raise ValueError(
            "Unknown data format. "
            "Supported formats are nifti and h5, got {}\n".format(data_config["format"])
        )

    labeled = data_config["labeled"]
    sample_label = "sample" if mode == "train" else "all"
    seed = None if mode == "train" else 0
    common_args = dict(
        file_loader=file_loader, labeled=labeled, sample_label=sample_label, seed=seed
    )

    if data_type == "paired":
        moving_image_shape = data_config["moving_image_shape"]
        fixed_image_shape = data_config["fixed_image_shape"]
        return PairedDataLoader(
            data_dir_path=os.path.join(data_dir, mode),
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
            data_dir_path=os.path.join(data_dir, mode),
            intra_group_prob=intra_group_prob,
            intra_group_option=intra_group_option,
            sample_image_in_group=sample_image_in_group,
            image_shape=image_shape,
            **common_args,
        )
    elif data_type == "unpaired":
        image_shape = data_config["image_shape"]
        return UnpairedDataLoader(
            data_dir_path=os.path.join(data_dir, mode),
            image_shape=image_shape,
            **common_args,
        )
    else:
        raise ValueError(
            "Unknown data format. "
            "Supported types are paired, unpaired, and grouped, got {}\n".format(
                data_type
            )
        )
