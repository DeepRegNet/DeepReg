# coding=utf-8

"""
Tests for deepreg/dataset/test_util.py in
pytest style
"""

import h5py
import numpy as np
import pytest
import tensorflow as tf
from testfixtures import TempDirectory

import deepreg.dataset.util as util
from deepreg.dataset.loader.interface import DataLoader
from deepreg.train import build_config
from deepreg.util import build_dataset


def test_sorted_h5_keys():
    """
    Test to check a key is returned with one entry
    """
    with TempDirectory() as tempdir:
        # Creating some dummy data
        d1 = np.random.random(size=(1000, 20))
        hf = h5py.File(tempdir.path + "data.h5", "w")
        hf.create_dataset("dataset_1", data=d1)
        hf.close()
        #  Checking func returns the same thing
        expected = ["dataset_1"]
        actual = util.get_h5_sorted_keys(tempdir.path + "data.h5")
        assert expected == actual


def test_sorted_h5_keys_many():
    """
    Test to check a key is returned with many entries
    """
    with TempDirectory() as tempdir:
        # Creating some dummy data
        d1 = np.random.random(size=(10, 20))
        hf = h5py.File(tempdir.path + "data.h5", "w")
        #  Adding entries in different order
        hf.create_dataset("dataset_1", data=d1)
        hf.create_dataset("dataset_3", data=d1)
        hf.create_dataset("dataset_2", data=d1)
        hf.close()
        #  Checking func returns the same thing
        expected = ["dataset_1", "dataset_2", "dataset_3"]
        actual = util.get_h5_sorted_keys(tempdir.path + "data.h5")
        assert expected == actual


def test_mkdirs_path_exists():
    """
    Testing case where path already exists
    """
    with TempDirectory() as tempdir:
        tempdir.makedir((tempdir.path + "/directory/"))
        util.mkdir_if_not_exists(tempdir.path + "/directory/")
        #  Checking correct dir structure
        tempdir.check_dir(tempdir.path + "/directory/")
        tempdir.cleanup()


def test_mkdirs_path_nonexistent():
    """
    Testing case where path doesn't exist
    """
    with TempDirectory() as tempdir:
        tempdir.makedir((tempdir.path + "/directory/"))
        util.mkdir_if_not_exists(tempdir.path + "/directory/new/")
        #  Checking new directory written
        tempdir.check_dir(tempdir.path + "/directory/new")
        tempdir.cleanup()


def test_get_sorted_filenames():
    """
    Checking sorted filenames returned by util function
    """
    with TempDirectory() as tempdir:
        tempdir.write((tempdir.path + "/a.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/b.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/c.txt"), (bytes(1)))
        expected = [
            tempdir.path + "/a.txt",
            tempdir.path + "/b.txt",
            tempdir.path + "/c.txt",
        ]
        actual = util.get_sorted_filenames_in_dir(tempdir.path, "txt")
        assert expected == actual


def test_difference_lists_same():
    """
    Check ValueError not raised if lists same
    """
    list_1 = list_2 = [0, 1, 2]
    util.check_difference_between_two_lists(list_1, list_2)


def test_difference_lists_different():
    """
    Assert ValueError raised if two lists not identical
    """
    list_1 = [0, 1, 2]
    list_2 = [3, 4, 5]
    with pytest.raises(ValueError):
        util.check_difference_between_two_lists(list_1, list_2)


def test_label_indices_sample():
    """
    Assert random number for passed arg returned
    """
    expected = set([0, 1, 2, 3])
    actual = util.get_label_indices(4, "sample")
    assert expected.intersection(set(actual))


def test_label_indices_first():
    """
    Assert list with 0 raised if first sample label
    """
    expected = [0]
    actual = util.get_label_indices(5, "first")
    assert expected == actual


def test_label_indices_all():
    """
    Assert list with all labels returned if all passed
    """
    expected = [0, 1, 2]
    actual = util.get_label_indices(3, "all")
    assert expected == actual


def test_label_indices_unknown():
    """
    Assert ValueError raised if unknown string passed to sample
    label
    """
    with pytest.raises(ValueError):
        util.get_label_indices(3, "random_str")


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
