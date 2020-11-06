#  coding=utf-8
"""
Tests functions in config/parser.py
"""

import os

import pytest
from testfixtures import TempDirectory

from deepreg.config import config_generator as cg


def test_gen_path_list():
    """
    Unit test checking list is returned from three
    paths
    """
    with TempDirectory() as d:
        #  Make new directories
        list_paths = ["train", "valid", "test"]
        list_made = [os.path.join(d.path, item) for item in list_paths]
        for item in list_paths:
            d.makedir(item)
        #  Test ValueError raised
        result = cg.gen_path_list(*list_made)
        assert all([item == result[i] for i, item in enumerate(list_made)])
        #  Cleanup directory
        d.cleanup()


#  parametrised test for value errors raised in gen_path_list
@pytest.mark.parametrize(
    "list_dirs",
    [([None, "valid", "test"]), (["train", None, "test"]), (["train", "valid", None])],
)
def test_gen_path_list_neg(list_dirs):
    """
    Function to test whether values errors raised when paths
    don't exist.
    """
    with TempDirectory() as d:
        #  Make new directories
        list_paths = []
        for item in list_dirs:
            if item:
                d.makedir(item)
                list_paths.append(item)
            else:
                list_paths.append("None")
        #  Test ValueError raised
        with pytest.raises(ValueError):
            cg.gen_path_list(list_paths[0], list_paths[1], list_paths[2])
        #  Cleanup directory
        d.cleanup()


def test_gen_path_one():
    """
    Unit test checking list is returned from one path
    """
    with TempDirectory() as d:
        #  Make new directories
        list_paths = ["train", "valid", "test"]
        list_made = [os.path.join(d.path, item) for item in list_paths]
        for item in list_paths:
            d.makedir(item)
        #  Test ValueError raised
        result = cg.gen_path_list_from_one_input(d.path)
        assert all([item == result[i] for i, item in enumerate(list_made)])
        #  Cleanup directory
        d.cleanup()


#  parametrised test for value errors raised in gen_path_list_from_one_input
@pytest.mark.parametrize(
    "list_dirs",
    [([None, "valid", "test"]), (["train", None, "test"]), (["train", "valid", None])],
)
def test_gen_path_list_one_neg(list_dirs):
    """
    Function to test whether values errors raised when paths
    don't exist.
    """
    with TempDirectory() as d:
        #  Make new directories
        list_paths = []
        for item in list_dirs:
            if item:
                d.makedir(item)
                list_paths.append(item)
            else:
                list_paths.append("None")
        #  Test ValueError raised
        with pytest.raises(ValueError):
            cg.gen_path_list_from_one_input(d.path)
        #  Cleanup directory
        d.cleanup()


#  Negative tests for gen_dataset_dict
@pytest.mark.parametrize(
    "dirs,format_im,type_loader,if_labeled,image_shape",
    [
        (["train", "valid", "test"], "nifti", "random_loader", True, [1, 1, 1]),
        (["train", "valid", "test"], "png", "unpaired", True, [1, 1, 1]),
        (["train", "valid", "test"], "nifti", "unpaired", True, [1, 1]),
        (["train", "valid", "test"], "nifti", "unpaired", True, [1, 1, 1, 1]),
        (["train", "valid", "test"], "nifti", "unpaired", True, [1, 1, "1"]),
        (["train", "valid", "test"], "nifti", "unpaired", True, [1, "1", 1]),
        (["train", "valid", "test"], "nifti", "unpaired", True, ["1", 1, 1]),
    ],
)
def test_gen_dataset_dict_neg(dirs, format_im, type_loader, if_labeled, image_shape):
    """
    Negative test to check value error raised with incorrect loader name
    or image shape
    """
    with pytest.raises(ValueError):
        cg.gen_dataset_dict(dirs, format_im, type_loader, if_labeled, image_shape)


def test_gen_dataset_dict_out():
    """
    Test output for gen_dataset_dict as expected.
    """
    #  Generate a dict
    args = [["train", "valid", "test"], "nifti", "unpaired", True, [1, 1, 1]]
    dict_out = cg.gen_dataset_dict(*args)
    dict_expect = {
        "dir": {"train": "train", "valid": "valid", "test": "test"},
        "format": "nifti",
        "type": "unpaired",
        "labeled": True,
        "image_shape": [1, 1, 1],
    }
    assert dict_out == dict_expect


#  Testing the optimizer dictionary generator
@pytest.mark.parametrize("optimizer,rate,momentum", [("random", 10.0, 0.0)])
def test_negative_optimizer_dict(optimizer, rate, momentum):
    """
    Check value error raised if incorrect optimizer used.
    """
    with pytest.raises(ValueError):
        cg.gen_optimizer_dict(optimizer, rate, momentum)


@pytest.mark.parametrize(
    "optimizer,rate,momentum,output",
    [
        (
            "sgd",
            10.0,
            None,
            {"name": "sgd", "sgd": {"learning_rate": 10.0, "momentum": 0.0}},
        ),
        (
            "sgd",
            10.0,
            0.1,
            {"name": "sgd", "sgd": {"learning_rate": 10.0, "momentum": 0.1}},
        ),
        (
            "adam",
            10.0,
            0.1,
            {"name": "adam", "adam": {"learning_rate": 10.0, "momentum": 0.1}},
        ),
        (
            "rms",
            10.0,
            0.1,
            {"name": "rms", "rms": {"learning_rate": 10.0, "momentum": 0.1}},
        ),
    ],
)
def test_optimizer_output(optimizer, rate, momentum, output):
    """
    Testing whetehr the optimizer dictionary generator returns
    the expected result.
    """
    if momentum:
        result = cg.gen_optimizer_dict(optimizer, rate, momentum)
    else:
        result = cg.gen_optimizer_dict(optimizer, rate)
    assert result == output
