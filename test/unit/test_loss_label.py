# coding=utf-8

"""
Tests for deepreg/model/loss/label.py in
pytest style
"""

from test.unit.util import is_equal_tf

import numpy as np
import pytest
import tensorflow as tf

import deepreg.loss.label as label


class TestMultiScaleLoss:
    def test_call(self):
        loss = label.MultiScaleLoss()
        with pytest.raises(NotImplementedError):
            loss.call(0, 0)

    def test_get_config(self):
        loss = label.MultiScaleLoss()
        got = loss.get_config()
        expected = dict(
            scales=None,
            kernel="gaussian",
            reduction=tf.keras.losses.Reduction.SUM,
            name="MultiScaleLoss",
        )
        assert got == expected


class TestDiceScore:
    shape = (3, 3, 3, 3)

    @pytest.fixture()
    def y_true(self):
        return np.ones(shape=self.shape) * 0.6

    @pytest.fixture()
    def y_pred(self):
        return np.ones(shape=self.shape) * 0.3

    @pytest.mark.parametrize(
        "binary,background_weight,scales,expected",
        [
            (True, 0.0, None, 0.0),
            (False, 0.0, None, 0.4),
            (False, 0.2, None, 0.4 / 0.94),
            (False, 0.2, [0, 0], 0.4 / 0.94),
            (False, 0.2, [0, 1], 0.46030036),
        ],
    )
    def test_call(self, y_true, y_pred, binary, background_weight, scales, expected):
        expected = np.array([expected] * self.shape[0])  # call returns (batch, )
        got = label.DiceScore(
            binary=binary, background_weight=background_weight, scales=scales
        ).call(y_true=y_true, y_pred=y_pred)
        assert is_equal_tf(got, expected)
        got = label.DiceLoss(
            binary=binary, background_weight=background_weight, scales=scales
        ).call(y_true=y_true, y_pred=y_pred)
        assert is_equal_tf(got, -expected)

    def test_get_config(self):
        got = label.DiceScore().get_config()
        expected = dict(
            binary=False,
            background_weight=0.0,
            scales=None,
            kernel="gaussian",
            reduction=tf.keras.losses.Reduction.SUM,
            name="DiceScore",
        )
        assert got == expected


class TestCrossEntropy:
    shape = (3, 3, 3, 3)

    @pytest.fixture()
    def y_true(self):
        return np.ones(shape=self.shape) * 0.6

    @pytest.fixture()
    def y_pred(self):
        return np.ones(shape=self.shape) * 0.3

    @pytest.mark.parametrize(
        "binary,background_weight,scales,expected",
        [
            (True, 0.0, None, -np.log(1.0e-7)),
            (False, 0.0, None, -0.6 * np.log(0.3)),
            (False, 0.2, None, -0.48 * np.log(0.3) - 0.08 * np.log(0.7)),
            (False, 0.2, [0, 0], -0.48 * np.log(0.3) - 0.08 * np.log(0.7)),
            (False, 0.2, [0, 1], 0.5239637),
        ],
    )
    def test_call(self, y_true, y_pred, binary, background_weight, scales, expected):
        expected = np.array([expected] * self.shape[0])  # call returns (batch, )
        got = label.CrossEntropy(
            binary=binary, background_weight=background_weight, scales=scales
        ).call(y_true=y_true, y_pred=y_pred)
        assert is_equal_tf(got, expected)

    def test_get_config(self):
        got = label.CrossEntropy().get_config()
        expected = dict(
            binary=False,
            background_weight=0.0,
            scales=None,
            kernel="gaussian",
            reduction=tf.keras.losses.Reduction.SUM,
            name="CrossEntropy",
        )
        assert got == expected


class TestJaccardIndex:
    shape = (3, 3, 3, 3)

    @pytest.fixture()
    def y_true(self):
        return np.ones(shape=self.shape) * 0.6

    @pytest.fixture()
    def y_pred(self):
        return np.ones(shape=self.shape) * 0.3

    @pytest.mark.parametrize(
        "binary,scales,expected",
        [
            (True, None, 0),
            (False, None, 0.25),
            (False, [0, 0], 0.25),
            (False, [0, 1], 0.17484076),
        ],
    )
    def test_call(self, y_true, y_pred, binary, scales, expected):
        expected = np.array([expected] * self.shape[0])  # call returns (batch, )
        got = label.JaccardIndex(binary=binary, scales=scales).call(
            y_true=y_true, y_pred=y_pred
        )
        assert is_equal_tf(got, expected)
        got = label.JaccardLoss(binary=binary, scales=scales).call(
            y_true=y_true, y_pred=y_pred
        )
        assert is_equal_tf(got, -expected)

    def test_get_config(self):
        got = label.JaccardIndex().get_config()
        expected = dict(
            binary=False,
            scales=None,
            kernel="gaussian",
            reduction=tf.keras.losses.Reduction.SUM,
            name="JaccardIndex",
        )
        assert got == expected


def test_foreground_prop_binary():
    """
    Test foreground function with a
    tensor of zeros with some ones, asserting
    equal to known precomputed tensor.
    Testing with binary case.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    expect = tf.convert_to_tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=tf.float32)
    get = label.foreground_proportion(tensor_eye)
    assert is_equal_tf(get, expect)


def test_foreground_prop_simple():
    """
    Test foreground functions with a tensor
    of zeros with some ones and some values below
    one to assert the thresholding works.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, 0, :, :] = 0.4 * array_eye  # 0
    tensor_eye[:, 1, :, :] = array_eye
    tensor_eye[:, 2, :, :] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)
    expect = [54 / (27 * 9), 54 / (27 * 9), 54 / (27 * 9)]
    get = label.foreground_proportion(tensor_eye)
    assert is_equal_tf(get, expect)


def test_compute_centroid():
    """
    Testing compute centroid function
    and comparing to expected values.
    """
    tensor_mask = np.zeros((3, 2, 2, 2))
    tensor_mask[0, :, :, :] = np.ones((2, 2, 2))
    tensor_mask = tf.constant(tensor_mask, dtype=tf.float32)

    tensor_grid = np.ones((1, 2, 2, 2, 3))
    tensor_grid[:, :, :, :, 1] *= 2
    tensor_grid[:, :, :, :, 2] *= 3
    tensor_grid = tf.constant(tensor_grid, dtype=tf.float32)

    expected = np.ones((3, 3))  # use 1 because 0/0 ~= (0+eps)/(0+eps) = 1
    expected[0, :] = [1, 2, 3]
    got = label.compute_centroid(tensor_mask, tensor_grid)
    assert is_equal_tf(got, expected)


def test_compute_centroid_d():
    """
    Testing compute centroid distance between equal
    tensors returns 0s.
    """
    array_ones = np.ones((2, 2))
    tensor_mask = np.zeros((3, 2, 2, 2))
    tensor_mask[0, :, :, :] = array_ones
    tensor_mask = tf.convert_to_tensor(tensor_mask, dtype=tf.float32)

    tensor_grid = np.zeros((1, 2, 2, 2, 3))
    tensor_grid[:, :, :, :, 0] = array_ones
    tensor_grid = tf.convert_to_tensor(tensor_grid, dtype=tf.float32)

    get = label.compute_centroid_distance(tensor_mask, tensor_mask, tensor_grid)
    expect = np.zeros((3))
    assert is_equal_tf(get, expect)
