# coding=utf-8
"""
Module predefining functions to add fields to a
.yaml config
"""
import os

# import yaml


# Functions for dataset fields
def gen_path_list(dir_train: str, dir_valid: str, dir_test: str):
    """
    Make a list out of the input dirs
    :param dir_train: dir where training data is stored
    :param dir_valid: dir where validation data is stored
    :param dir_test: dir where test data is stored
    """
    return [dir_train, dir_valid, dir_test]


def gen_path_list_from_one_input(root_dir: str):
    """
    Function that generates paths for
    training, testing, validation sets provided
    a root directory and assuming the structure of
    folders with train, valid, test subfolders.
    :param root_dir: top level directory that
    """
    #  Autogen the folders from correct structure
    dir_train = os.path.join(root_dir, "train")
    dir_valid = os.path.join(root_dir, "valid")
    dir_test = os.path.join(root_dir, "test")

    #  We need to check if users have set up their folders in the right
    #  structure
    if not os.path.exists(dir_train):
        raise ValueError("Train folder path: {} does not exist.".format(dir_train))
    if not os.path.exists(dir_valid):
        raise ValueError("Valid folder path: {} does not exist.".format(dir_valid))
    if not os.path.exists(dir_test):
        raise ValueError("Test folder path: {} does not exist.".format(dir_test))

    #  If folders exist in root dir, return the list of folders
    return [dir_train, dir_valid, dir_test]


def gen_dataset_dict(
    dirs: list, format_im: str, type_loader: str, if_labeled: bool, image_shape: list
):
    """
    Function which creates the required dictionary
    structure from inputs for the .yaml config file
    :param dirs: list of directories associated to train, valid
                and test folders
    :param format_im: string, denoting data format, one of (?)
    :param type_loader: string, denoting type of data loader, one of (paired,
                 unpaired, grouped)
    :param if_labeled: Bool, whether to use labels, if available.
    :param image_shape: list of ints, dimensions of input network.
    """
    #  Asserting inputs
    # Checking type_loadr
    if type_loader not in ["unpaired", "paired", "grouped"]:
        raise ValueError("Unsupported dataloader: {}".format(type_loader))

    # Checking format ims TODO

    #  Checking image_format
    if len(image_shape) != 3:
        raise ValueError(
            "Currently only support 3D images, input shape: {}".format(len(image_shape))
        )

    if not all([isinstance(item, int) for item in image_shape]):
        raise ValueError("One of the image dims not int: {}".format(*image_shape))

    #  Define storage dictionary.
    dataset_dict = dict()

    # Adding dir field
    dataset_dict["dir"] = dict()
    dataset_dict["dir"]["train"] = dirs[0]
    dataset_dict["dir"]["valid"] = dirs[1]
    dataset_dict["dir"]["test"] = dirs[2]
    dataset_dict["format"] = format_im
    dataset_dict["type"] = type_loader
    dataset_dict["labeled"] = if_labeled
    dataset_dict["image_shape"] = image_shape

    return dataset_dict


#  Functions for train fields

#  Functions for loss fields

#  Functions for optimizer fields
def gen_optimizer_dict(optimizer: str, rate: float, momentum: float):
    """
    Function which creates the required dictionary
    structure from inputs for the .yaml config file
    :param optimizer: which optimizer to use ("adam"|"sgd"|"rms")
    :param rate: what learning rate to use for the optimizer
    :param momentum: what momentum to use for optimizer, only
                     applicable to sgd, rms
    """
    #  Check inputs are correct
    if optimizer not in ["adam", "sgd", "rms"]:
        raise ValueError(
            "Unsupported optimiser: {}, must be one of adam, sgd, rms".format(optimizer)
        )

    #  Add to dictionary
    dict_optimizer = dict()
    dict_optimizer["name"] = optimizer
    dict_optimizer[optimizer] = dict()
    dict_optimizer[optimizer]["learning_rate"] = rate
    if optimizer != "adam":
        dict_optimizer[optimizer]["momentum"] = momentum

    return dict_optimizer


#  FUnctions for preprocessing fields

#  Functions for other hyperparams

# Write a yaml file to a path given input dictionary
