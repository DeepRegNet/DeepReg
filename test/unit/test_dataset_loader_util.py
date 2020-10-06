# coding=utf-8

"""
Tests for deepreg/dataset/loader/util.py in pytest style
"""

from test.unit.util import is_equal_np

import numpy as np
import pytest

import deepreg.dataset.loader.util as util


class TestNormalizeArray:
    @pytest.mark.parametrize(
        "arr,expected",
        [
            [np.array([0, 1, 2]), np.array([0, 0.5, 1])],
            [np.array([-2, 0, 1, 2]), np.array([0, 0.5, 0.75, 1])],
            [np.array([1, 1, 1, 1]), np.array([0, 0, 0, 0])],
        ],
    )
    def test_no_min_no_max(self, arr, expected):
        got = util.normalize_array(arr=arr)
        assert is_equal_np(got, expected)

    @pytest.mark.parametrize(
        "arr,v_max,expected",
        [
            [np.array([0, 1, 2]), 1, np.array([0, 1, 1])],
            [np.array([-2, 0, 1, 2]), 3, np.array([0, 0.4, 0.6, 0.8])],
        ],
    )
    def test_no_min(self, arr, v_max, expected):
        got = util.normalize_array(arr=arr, v_max=v_max)
        assert is_equal_np(got, expected)

    @pytest.mark.parametrize(
        "arr,v_min,expected",
        [
            [np.array([0, 1, 2]), 1, np.array([0, 0, 1])],
            [np.array([-2, 0, 1, 2]), -3, np.array([0.2, 0.6, 0.8, 1])],
        ],
    )
    def test_no_max(self, arr, v_min, expected):
        got = util.normalize_array(arr=arr, v_min=v_min)
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
