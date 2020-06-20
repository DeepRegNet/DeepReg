import os

from deepreg.data.nifti.nifti_paired_labeled_loader import NiftiPairedLabeledDataLoader
from deepreg.data.nifti.nifti_paired_unlabeled_loader import NiftiPairedUnlabeledDataLoader
from deepreg.data.nifti.nifti_unpaired_loader import NiftiUnpairedLabeledDataLoader


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
        raise ValueError("training data must be provided, they should be stored under train/")
    if mode == "valid" and mode not in modes:
        # when validation data is not available, use test data instead
        mode = "test"
    if mode not in modes:
        return None

    sample_label = "sample" if mode == "train" else "all"
    seed = None if mode == "train" else 0

    if data_config["format"] == "nifti":
        if data_config["paired"] is True:
            moving_image_shape = data_config["moving_image_shape"]
            fixed_image_shape = data_config["fixed_image_shape"]
            if data_config["labeled"] is True:
                return NiftiPairedLabeledDataLoader(data_dir_path=os.path.join(data_dir, mode),
                                                    sample_label=sample_label,
                                                    seed=seed,
                                                    moving_image_shape=moving_image_shape,
                                                    fixed_image_shape=fixed_image_shape)
            else:
                return NiftiPairedUnlabeledDataLoader(data_dir_path=os.path.join(data_dir, mode),
                                                      sample_label=None,
                                                      seed=seed,
                                                      moving_image_shape=moving_image_shape,
                                                      fixed_image_shape=fixed_image_shape)
        elif data_config["paired"] is False:
            if data_config["labeled"] is True:
                image_shape = data_config["image_shape"]
                return NiftiUnpairedLabeledDataLoader(data_dir_path=os.path.join(data_dir, mode),
                                                      sample_label=sample_label,
                                                      seed=seed,
                                                      image_shape=image_shape)
    raise ValueError("Unknown data loader type. \nConfig {}\n".format(data_config))
