# coding=utf-8

"""
Tests for deepreg/model/loss/label.py in
pytest style
"""

from test.unit.util import is_equal_tf
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

import deepreg.loss.label as label
from deepreg.constant import EPS


class TestSumSquaredDistance:
    @pytest.mark.parametrize(
        "y_true,y_pred,shape,expected",
        [
            (0.6, 0.3, (3,), 0.09),
            (0.6, 0.3, (3, 3), 0.09),
            (0.6, 0.3, (3, 3, 3), 0.09),
            (0.6, 0.3, (3, 3, 3), 0.09),
            (0.5, 0.5, (3, 3), 0.0),
            (0.3, 0.6, (3, 3), 0.09),
        ],
    )
    def test_output(self, y_true, y_pred, shape, expected):
        y_true = y_true * np.ones(shape=shape)
        y_pred = y_pred * np.ones(shape=shape)
        expected = expected * np.ones(shape=(shape[0],))
        got = label.SumSquaredDifference().call(
            y_true,
            y_pred,
        )
        assert is_equal_tf(got, expected)


class TestDiceScore:
    @pytest.mark.parametrize(
        ("value", "smooth_nr", "smooth_dr", "expected"),
        [
            (0, 1e-5, 1e-5, 1),
            (0, 0, 1e-5, 0),
            (0, 1e-5, 0, np.inf),
            (0, 0, 0, np.nan),
            (0, 1e-7, 1e-7, 1),
            (1, 1e-5, 1e-5, 1),
            (1, 0, 1e-5, 1),
            (1, 1e-5, 0, 1),
            (1, 0, 0, 1),
            (1, 1e-7, 1e-7, 1),
        ],
    )
    def test_smooth(
        self,
        value: float,
        smooth_nr: float,
        smooth_dr: float,
        expected: float,
    ):
        """
        Test values in extreme cases where numerator/denominator are all zero.

        :param value: value for input.
        :param smooth_nr: constant for numerator.
        :param smooth_dr: constant for denominator.
        :param expected: target value.
        """
        shape = (1, 10)
        y_true = tf.ones(shape=shape) * value
        y_pred = tf.ones(shape=shape) * value

        got = label.DiceScore(smooth_nr=smooth_nr, smooth_dr=smooth_dr).call(
            y_true,
            y_pred,
        )
        expected = tf.constant(expected)
        assert is_equal_tf(got[0], expected)

    @pytest.mark.parametrize("binary", [True, False])
    @pytest.mark.parametrize("background_weight", [0.0, 0.1, 0.5, 1.0])
    @pytest.mark.parametrize("shape", [(1,), (10,), (100,), (2, 3), (2, 3, 4)])
    def test_exact_value(self, binary: bool, background_weight: float, shape: Tuple):
        """
        Test dice score by comparing at ground truth values.

        :param binary: if project labels to binary values.
        :param background_weight: the weight of background class.
        :param shape: shape of input.
        """
        # init
        shape = (1,) + shape  # add batch axis
        foreground_weight = 1 - background_weight
        tf.random.set_seed(0)
        y_true = tf.random.uniform(shape=shape)
        y_pred = tf.random.uniform(shape=shape)

        # obtained value
        got = label.DiceScore(
            binary=binary,
            background_weight=background_weight,
        ).call(y_true=y_true, y_pred=y_pred)

        # expected value
        flatten = tf.keras.layers.Flatten()
        y_true = flatten(y_true)
        y_pred = flatten(y_pred)
        if binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        num = foreground_weight * tf.reduce_sum(
            y_true * y_pred, axis=1
        ) + background_weight * tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=1)
        num *= 2
        denom = foreground_weight * tf.reduce_sum(
            y_true + y_pred, axis=1
        ) + background_weight * tf.reduce_sum((1 - y_true) + (1 - y_pred), axis=1)
        expected = (num + EPS) / (denom + EPS)

        assert is_equal_tf(got, expected)

    def test_get_config(self):
        got = label.DiceScore().get_config()
        expected = dict(
            binary=False,
            background_weight=0.0,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            reduction=tf.keras.losses.Reduction.AUTO,
            name="DiceScore",
        )
        assert got == expected

    @pytest.mark.parametrize("background_weight", [-0.1, 1.1])
    def test_background_weight_err(self, background_weight: float):
        """
        Test the error message when using wrong background weight.

        :param background_weight: weight for background class.
        """
        with pytest.raises(ValueError) as err_info:
            label.DiceScore(background_weight=background_weight)
        assert "The background weight for Dice Score must be within [0, 1]" in str(
            err_info.value
        )


class TestCrossEntropy:
    shape = (3, 3, 3, 3)

    @pytest.fixture()
    def y_true(self):
        return np.ones(shape=self.shape) * 0.6

    @pytest.fixture()
    def y_pred(self):
        return np.ones(shape=self.shape) * 0.3

    @pytest.mark.parametrize(
        ("value", "smooth", "expected"),
        [
            (0, 1e-5, 0),
            (0, 0, np.nan),
            (0, 1e-7, 0),
            (1, 1e-5, -np.log(1 + 1e-5)),
            (1, 0, 0),
            (1, 1e-7, -np.log(1 + 1e-7)),
        ],
    )
    def test_smooth(
        self,
        value: float,
        smooth: float,
        expected: float,
    ):
        """
        Test values in extreme cases where numerator/denominator are all zero.

        :param value: value for input.
        :param smooth: constant for log.
        :param expected: target value.
        """
        shape = (1, 10)
        y_true = tf.ones(shape=shape) * value
        y_pred = tf.ones(shape=shape) * value

        got = label.CrossEntropy(smooth=smooth).call(
            y_true,
            y_pred,
        )
        expected = tf.constant(expected)
        assert is_equal_tf(got[0], expected)

    @pytest.mark.parametrize(
        "binary,background_weight,expected",
        [
            (True, 0.0, -np.log(EPS)),
            (False, 0.0, -0.6 * np.log(0.3 + EPS)),
            (False, 0.2, -0.48 * np.log(0.3 + EPS) - 0.08 * np.log(0.7 + EPS)),
        ],
    )
    def testcall(self, y_true, y_pred, binary, background_weight, expected):
        expected = np.array([expected] * self.shape[0])  # call returns (batch, )
        got = label.CrossEntropy(
            binary=binary,
            background_weight=background_weight,
        ).call(y_true=y_true, y_pred=y_pred)
        assert is_equal_tf(got, expected)

    def test_get_config(self):
        got = label.CrossEntropy().get_config()
        expected = dict(
            binary=False,
            background_weight=0.0,
            smooth=1e-5,
            reduction=tf.keras.losses.Reduction.AUTO,
            name="CrossEntropy",
        )
        assert got == expected

    @pytest.mark.parametrize("background_weight", [-0.1, 1.1])
    def test_background_weight_err(self, background_weight: float):
        """
        Test the error message when using wrong background weight.

        :param background_weight: weight for background class.
        """
        with pytest.raises(ValueError) as err_info:
            label.CrossEntropy(background_weight=background_weight)
        assert "The background weight for Cross Entropy must be within [0, 1]" in str(
            err_info.value
        )


class TestJaccardIndex:
    @pytest.mark.parametrize(
        ("value", "smooth_nr", "smooth_dr", "expected"),
        [
            (0, 1e-5, 1e-5, 1),
            (0, 0, 1e-5, 0),
            (0, 1e-5, 0, np.inf),
            (0, 0, 0, np.nan),
            (0, 1e-7, 1e-7, 1),
            (1, 1e-5, 1e-5, 1),
            (1, 0, 1e-5, 1),
            (1, 1e-5, 0, 1),
            (1, 0, 0, 1),
            (1, 1e-7, 1e-7, 1),
        ],
    )
    def test_smooth(
        self,
        value: float,
        smooth_nr: float,
        smooth_dr: float,
        expected: float,
    ):
        """
        Test values in extreme cases where numerator/denominator are all zero.

        :param value: value for input.
        :param smooth_nr: constant for numerator.
        :param smooth_dr: constant for denominator.
        :param expected: target value.
        """
        shape = (1, 10)
        y_true = tf.ones(shape=shape) * value
        y_pred = tf.ones(shape=shape) * value

        got = label.JaccardIndex(smooth_nr=smooth_nr, smooth_dr=smooth_dr).call(
            y_true,
            y_pred,
        )
        expected = tf.constant(expected)
        assert is_equal_tf(got[0], expected)

    @pytest.mark.parametrize("binary", [True, False])
    @pytest.mark.parametrize("background_weight", [0.0, 0.1, 0.5, 1.0])
    @pytest.mark.parametrize("shape", [(1,), (10,), (100,), (2, 3), (2, 3, 4)])
    def test_exact_value(self, binary: bool, background_weight: float, shape: Tuple):
        """
        Test Jaccard index by comparing at ground truth values.

        :param binary: if project labels to binary values.
        :param background_weight: the weight of background class.
        :param shape: shape of input.
        """
        # init
        shape = (1,) + shape  # add batch axis
        foreground_weight = 1 - background_weight
        tf.random.set_seed(0)
        y_true = tf.random.uniform(shape=shape)
        y_pred = tf.random.uniform(shape=shape)

        # obtained value
        got = label.JaccardIndex(
            binary=binary,
            background_weight=background_weight,
        ).call(y_true=y_true, y_pred=y_pred)

        # expected value
        flatten = tf.keras.layers.Flatten()
        y_true = flatten(y_true)
        y_pred = flatten(y_pred)
        if binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        num = foreground_weight * tf.reduce_sum(
            y_true * y_pred, axis=1
        ) + background_weight * tf.reduce_sum((1 - y_true) * (1 - y_pred), axis=1)
        denom = foreground_weight * tf.reduce_sum(
            y_true + y_pred, axis=1
        ) + background_weight * tf.reduce_sum((1 - y_true) + (1 - y_pred), axis=1)
        denom = denom - num
        expected = (num + EPS) / (denom + EPS)

        assert is_equal_tf(got, expected)

    def test_get_config(self):
        got = label.JaccardIndex().get_config()
        expected = dict(
            binary=False,
            background_weight=0.0,
            smooth_nr=1e-5,
            smooth_dr=1e-5,
            reduction=tf.keras.losses.Reduction.AUTO,
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
