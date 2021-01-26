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
        "binary,neg_weight,scales,expected",
        [
            (True, 0.0, None, 0.0),
            (False, 0.0, None, 0.4),
            (False, 0.2, None, 0.4 / 0.94),
            (False, 0.2, [0, 0], 0.4 / 0.94),
            (False, 0.2, [0, 1], 0.46030036),
        ],
    )
    def test_call(self, y_true, y_pred, binary, neg_weight, scales, expected):
        expected = np.array([expected] * self.shape[0])  # call returns (batch, )
        got = label.DiceScore(binary=binary, neg_weight=neg_weight, scales=scales).call(
            y_true=y_true, y_pred=y_pred
        )
        assert is_equal_tf(got, expected)
        got = label.DiceLoss(binary=binary, neg_weight=neg_weight, scales=scales).call(
            y_true=y_true, y_pred=y_pred
        )
        assert is_equal_tf(got, -expected)

    def test_get_config(self):
        got = label.DiceScore().get_config()
        expected = dict(
            binary=False,
            neg_weight=0.0,
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
        "binary,neg_weight,scales,expected",
        [
            (True, 0.0, None, -np.log(1.0e-7)),
            (False, 0.0, None, -0.6 * np.log(0.3)),
            (False, 0.2, None, -0.48 * np.log(0.3) - 0.08 * np.log(0.7)),
            (False, 0.2, [0, 0], -0.48 * np.log(0.3) - 0.08 * np.log(0.7)),
            (False, 0.2, [0, 1], 0.5239637),
        ],
    )
    def test_call(self, y_true, y_pred, binary, neg_weight, scales, expected):
        expected = np.array([expected] * self.shape[0])  # call returns (batch, )
        got = label.CrossEntropy(
            binary=binary, neg_weight=neg_weight, scales=scales
        ).call(y_true=y_true, y_pred=y_pred)
        assert is_equal_tf(got, expected)

    def test_get_config(self):
        got = label.CrossEntropy().get_config()
        expected = dict(
            binary=False,
            neg_weight=0.0,
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
