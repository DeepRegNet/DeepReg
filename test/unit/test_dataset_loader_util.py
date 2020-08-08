# coding=utf-8

"""
Tests for deepreg/dataset/loader/util.py in pytest style
"""

from test.unit.util import is_equal_np

import numpy as np

import deepreg.dataset.loader.util as util


def test_normalize_array():
    """
    Test normalize array by checking the output values
    """

    # no v_min, v_max
    arr = np.array([0, 1, 2])
    got = util.normalize_array(arr=arr)
    expected = np.array([0, 0.5, 1])
    assert is_equal_np(got, expected)

    arr = np.array([-2, 0, 1, 2])
    got = util.normalize_array(arr=arr)
    expected = np.array([0, 0.5, 0.75, 1])
    assert is_equal_np(got, expected)

    # no v_min
    arr = np.array([0, 1, 2])
    got = util.normalize_array(arr=arr, v_max=1)
    expected = np.array([0, 1, 1])
    assert is_equal_np(got, expected)

    arr = np.array([-2, 0, 1, 2])
    got = util.normalize_array(arr=arr, v_max=3)
    expected = np.array([0, 0.4, 0.6, 0.8])
    assert is_equal_np(got, expected)

    # no v_max
    arr = np.array([0, 1, 2])
    got = util.normalize_array(arr=arr, v_min=1)
    expected = np.array([0, 0, 1])
    assert is_equal_np(got, expected)

    arr = np.array([-2, 0, 1, 2])
    got = util.normalize_array(arr=arr, v_min=-3)
    expected = np.array([0.2, 0.6, 0.8, 1])
    assert is_equal_np(got, expected)


def test_remove_prefix_suffix():
    """
    Test remove_prefix_suffix by verifying outputs
    """
    x = "sample000.nii"

    # single prefix, suffix
    got = util.remove_prefix_suffix(x=x, prefix="sample", suffix=".nii")
    expected = "000"
    assert got == expected

    # multiple prefixes, suffixes
    got = util.remove_prefix_suffix(x=x, prefix=["sample"], suffix=[".nii.gz", ".nii"])
    expected = "000"
    assert got == expected
