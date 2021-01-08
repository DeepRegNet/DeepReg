# coding=utf-8

"""
Tests for deepreg/dataset/util.py in
pytest style
"""

import h5py
import numpy as np
import pytest
from testfixtures import TempDirectory

import deepreg.dataset.util as util


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
        # Checking func returns the same thing
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
        # Adding entries in different order
        hf.create_dataset("dataset_1", data=d1)
        hf.create_dataset("dataset_3", data=d1)
        hf.create_dataset("dataset_2", data=d1)
        hf.close()
        # Checking func returns the same thing
        expected = ["dataset_1", "dataset_2", "dataset_3"]
        actual = util.get_h5_sorted_keys(tempdir.path + "data.h5")
        assert expected == actual


def test_get_sorted_file_paths_in_dir_with_suffix():
    """
    Checking sorted file names returned
    """

    # one dir, single suffix
    with TempDirectory() as tempdir:
        tempdir.write((tempdir.path + "/a.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/b.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/c.txt"), (bytes(1)))
        expected = [("a", "txt"), ("b", "txt"), ("c", "txt")]
        actual = util.get_sorted_file_paths_in_dir_with_suffix(tempdir.path, "txt")
        assert expected == actual

    # one dir, multiple suffixes
    with TempDirectory() as tempdir:
        tempdir.write((tempdir.path + "/a.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/b.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/c.md"), (bytes(1)))
        expected = [("a", "txt"), ("b", "txt"), ("c", "md")]
        actual = util.get_sorted_file_paths_in_dir_with_suffix(
            tempdir.path, ["txt", "md"]
        )
        assert expected == actual

    # multiple sub-dirs, single suffix
    with TempDirectory() as tempdir:
        tempdir.write((tempdir.path + "/1/a.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/1/b.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/1/c.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/2/a.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/2/b.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/2/c.txt"), (bytes(1)))
        expected = [
            ("1/a", "txt"),
            ("1/b", "txt"),
            ("1/c", "txt"),
            ("2/a", "txt"),
            ("2/b", "txt"),
            ("2/c", "txt"),
        ]
        actual = util.get_sorted_file_paths_in_dir_with_suffix(tempdir.path, "txt")
        assert expected == actual

    # multiple sub-dirs, multiple suffixes
    with TempDirectory() as tempdir:
        tempdir.write((tempdir.path + "/1/a.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/1/b.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/1/c.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/2/a.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/2/b.txt"), (bytes(1)))
        tempdir.write((tempdir.path + "/2/c.md"), (bytes(1)))
        expected = [
            ("1/a", "txt"),
            ("1/b", "txt"),
            ("1/c", "txt"),
            ("2/a", "txt"),
            ("2/b", "txt"),
            ("2/c", "md"),
        ]
        actual = util.get_sorted_file_paths_in_dir_with_suffix(
            tempdir.path, ["txt", "md"]
        )
        assert expected == actual


def test_check_difference_between_two_lists():
    """
    Check check_difference_between_two_lists by verifying ValueError
    """
    # same lists, no error
    list1 = list2 = [0, 1, 2]
    util.check_difference_between_two_lists(list1, list2, name="same case")

    # diff lists with same unique numbers
    list_1 = [0, 1, 2]
    list_2 = [1, 2, 0]
    with pytest.raises(ValueError) as err_info:
        util.check_difference_between_two_lists(list_1, list_2, name="diff case")
    assert "diff case are not identical" in str(err_info.value)

    # totally diff lists
    list_1 = [0, 1, 2]
    list_2 = [3, 4, 5]
    with pytest.raises(ValueError) as err_info:
        util.check_difference_between_two_lists(list_1, list_2, name="diff case")
    assert "diff case are not identical" in str(err_info.value)


def test_label_indices_sample():
    """
    Assert random number for passed arg returned
    """
    expected = {0, 1, 2, 3}
    actual = util.get_label_indices(4, "sample")
    assert expected.intersection(set(actual))


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
