import os
import re

import tensorflow as tf

from deepreg.dataset.loader.interface import DataLoader
from deepreg.train import build_config
from deepreg.util import build_dataset, build_log_dir


def test_build_dataset():
    """
    Test build_dataset by checking the output types
    """

    # init arguments
    config_path = "deepreg/config/unpaired_labeled_ddf.yaml"
    log_dir = "test_build_dataset"
    ckpt_path = ""

    # load config
    config, log_dir = build_config(
        config_path=config_path, log_dir=log_dir, ckpt_path=ckpt_path
    )

    # build dataset
    data_loader_train, dataset_train, steps_per_epoch_train = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        mode="train",
    )

    # check output types
    assert isinstance(data_loader_train, DataLoader)
    assert isinstance(dataset_train, tf.data.Dataset)
    assert isinstance(steps_per_epoch_train, int)

    # remove valid data
    config["dataset"]["dir"]["valid"] = ""

    # build dataset
    data_loader_valid, dataset_valid, steps_per_epoch_valid = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        mode="valid",
    )

    assert data_loader_valid is None
    assert dataset_valid is None
    assert steps_per_epoch_valid is None


def test_build_log_dir():
    """
    Test build_log_dir for default directory and custom directory
    """

    # use default timestamp based directory
    log_dir = build_log_dir(log_dir="")
    head, tail = os.path.split(log_dir)
    assert head == "logs"
    pattern = re.compile("[0-9]{8}-[0-9]{6}")
    assert pattern.match(tail)

    # use custom directory
    log_dir = build_log_dir(log_dir="custom")
    head, tail = os.path.split(log_dir)
    assert head == "logs"
    assert tail == "custom"
