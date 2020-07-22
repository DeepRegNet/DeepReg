import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from deepreg.dataset.load import get_data_loader
from deepreg.dataset.loader.interface import DataLoader


def build_dataset(
    dataset_config: dict,
    preprocess_config: dict,
    mode: str,
    training: bool,
    repeat: bool,
) -> [(DataLoader, None), (tf.data.Dataset, None), (int, None)]:
    """
    Function to prepare dataset for training and validation.
    :param dataset_config: configuration for dataset
    :param preprocess_config: configuration for preprocess
    :param mode: train or valid or test
    :param training: bool, if true, data augmentation and shuffling will be added
    :param repeat: bool, if true, dataset will be repeated, true for train/valid dataset during model.fit
    :return:
    - (data_loader_train, dataset_train, steps_per_epoch_train)
    - (data_loader_val, dataset_val, steps_per_epoch_valid)

    Cannot move this function into deepreg/dataset/util.py
    as we need DataLoader to define the output
    """
    assert mode in ["train", "valid", "test"]
    data_loader = get_data_loader(dataset_config, mode)
    if data_loader is None:
        return None, None, None
    dataset = data_loader.get_dataset_and_preprocess(
        training=training, repeat=repeat, **preprocess_config
    )
    dataset_size = data_loader.num_samples
    steps_per_epoch = max(dataset_size // preprocess_config["batch_size"], 1)
    return data_loader, dataset, steps_per_epoch


def build_log_dir(log_dir: str) -> str:
    """
    :param log_dir: str, path to where training logs to be stored.
    :return: the path of directory to save logs
    """
    log_dir = os.path.join(
        "logs", datetime.now().strftime("%Y%m%d-%H%M%S") if log_dir == "" else log_dir
    )
    if os.path.exists(log_dir):
        logging.warning("Log directory {} exists already.".format(log_dir))
    else:
        os.makedirs(log_dir)
    return log_dir


def save_array(sample_dir: str, arr: np.ndarray, prefix: str, name: str, gray: bool):
    """
    :param sample_dir: path of the directory to save
    :param arr: 3D or 4D array to be saved
    :param prefix: "moving" or "fixed"
    :param name: name of the array, e.g. image, label, etc.
    :param gray: true if the array is between 0,1
    """
    assert len(arr.shape) in [3, 4]
    is_4d = len(arr.shape) == 4
    if is_4d:
        # if 4D array, it must be 3 channels
        assert arr.shape[3] == 3
    assert prefix in ["fixed", "moving"]

    for depth_index in range(arr.shape[2]):
        plt.imsave(
            fname=os.path.join(
                f"{sample_dir}", f"{prefix}_depth{depth_index}_{name}.png"
            ),
            arr=arr[:, :, depth_index, :] if is_4d else arr[:, :, depth_index],
            vmin=0 if gray else None,
            vmax=1 if gray else None,
            cmap="gray" if gray else "PiYG",
        )


def get_mean_median_std(values):
    return np.mean(values), np.median(values), np.std(values)
