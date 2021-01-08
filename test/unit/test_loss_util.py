# coding=utf-8

"""
Tests for deepreg/model/loss/label.py in
pytest style
"""

from test.unit.util import is_equal_tf

import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.loss.label as label


@pytest.mark.parametrize("sigma", [1, 3, 2.2])
def test_gaussian_kernel1d(sigma):
    tail = int(sigma * 3)
    expected = [np.exp(-0.5 * x ** 2 / sigma ** 2) for x in range(-tail, tail + 1)]
    expected = expected / np.sum(expected)
    got = label.gaussian_kernel1d(sigma)
    assert is_equal_tf(got, expected)


@pytest.mark.parametrize("sigma", [1, 3, 2.2])
def test_cauchy_kernel1d(sigma):
    tail = int(sigma * 5)
    expected = [1 / ((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)]
    expected = expected / np.sum(expected)
    got = label.cauchy_kernel1d(sigma)
    assert is_equal_tf(got, expected)


def test_separable_filter():
    """
    Testing separable filter case where non
    zero length tensor is passed to the
    function.
    """
    k = np.ones((3, 3, 3, 3, 1), dtype=np.float32)
    array_eye = np.identity(3, dtype=np.float32)
    tensor_pred = np.zeros((3, 3, 3, 3, 1), dtype=np.float32)
    tensor_pred[:, :, 0, 0, 0] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)
    k = tf.convert_to_tensor(k, dtype=tf.float32)

    expect = np.ones((3, 3, 3, 3, 1), dtype=np.float32)
    expect = tf.convert_to_tensor(expect, dtype=tf.float32)

    get = label.separable_filter(tensor_pred, k)
    assert is_equal_tf(get, expect)
