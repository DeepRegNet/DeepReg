import logging
import os
from datetime import datetime

import tensorflow as tf

from deepreg.dataset.load import get_data_loader
from deepreg.dataset.loader.interface import DataLoader


def build_dataset(
    dataset_config: dict, preprocess_config: dict, mode: str
) -> [(DataLoader, None), (tf.data.Dataset, None), (int, None)]:
    """
    Function to prepare dataset for training and validation.
    :param dataset_config: configuration for dataset
    :param preprocess_config: configuration for preprocess
    :param mode: train or valid or test
    :return:
    - (data_loader_train, dataset_train, steps_per_epoch_train)
    - (data_loader_val, dataset_val, steps_per_epoch_valid)
    """
    assert mode in ["train", "valid", "test"]
    data_loader = get_data_loader(dataset_config, mode)
    if data_loader is None:
        return None, None, None
    dataset = data_loader.get_dataset_and_preprocess(
        training=mode == "train", repeat=True, **preprocess_config
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
