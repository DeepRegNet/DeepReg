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


# def test_gen_path_list_neg():
#     """
#     Check assertion errors raised if the paths don't exist
#     """
