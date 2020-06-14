import os

from deepreg.data.nifti.nifti_paired_loader import NiftiPairedDataLoader


def get_data_loader(data_config, mode):
    """
    Return the corresponding data loader.
    Can't be placed in the same file of loader interfaces as it causes import cycle.
    :param data_config:
    :param mode:
    :return:
    """
    data_dir = data_config["dir"]
    modes = os.listdir(data_dir)
    if "train" not in modes:
        raise ValueError("training data must be provided, they should be stored under train/")
    if mode == "valid" and mode not in modes:
        # when validation data is not available, use test data instead
        mode = "test"
    if mode not in modes:
        return None

    moving_image_shape = data_config["moving_image_shape"]
    fixed_image_shape = data_config["fixed_image_shape"]
    if data_config["type"] == "nifti":
        sample_label = data_config["sample_label"][mode]
        return NiftiPairedDataLoader(data_dir_path=os.path.join(data_dir, mode),
                                     moving_image_shape=moving_image_shape,
                                     fixed_image_shape=fixed_image_shape,
                                     sample_label=sample_label)
    else:
        raise ValueError("Unknown data loader type")
