# coding=utf-8

"""
Tests for deepreg/model/loss/label.py in
pytest style
"""
from test.unit.util import is_equal_tf
from typing import List, Optional, Union

import numpy as np
import pytest
import tensorflow as tf

from deepreg.loss.label import DiceLoss, DiceScore
from deepreg.loss.util import MultiScaleMixin, separable_filter


class TestMultiScaleMixin:
    def test_err(self):
        with pytest.raises(ValueError) as err_info:
            MultiScaleMixin(kernel="unknown")
        assert "Kernel unknown is not supported." in str(err_info.value)

    def test_get_config(self):
        loss = MultiScaleMixin()
        got = loss.get_config()
        expected = dict(
            scales=None,
            kernel="gaussian",
            reduction=tf.keras.losses.Reduction.AUTO,
            name=None,
        )
        assert got == expected

    @pytest.mark.parametrize("kernel", ["gaussian", "cauchy"])
    @pytest.mark.parametrize("scales", [None, 0, [0], [0, 1], [1, 2]])
    def test_call(self, kernel: str, scales: Optional[Union[List, float, int]]):
        """
        Test MultiScaleMixin using DiceLoss.

        :param kernel: kernel name.
        :param scales: scaling parameters.
        """
        shape = (2, 3, 4, 5)
        y_true = tf.random.uniform(shape=shape)
        y_pred = tf.random.uniform(shape=shape)

        loss = DiceLoss(kernel=kernel, scales=scales)
        loss.call(y_pred=y_pred, y_true=y_true)


def test_negative_loss_mixin():
    """Test DiceScore and DiceLoss have reversed sign."""
    shape = (2, 3, 4, 5)
    y_true = tf.random.uniform(shape=shape)
    y_pred = tf.random.uniform(shape=shape)

    dice_score = DiceScore().call(y_pred=y_pred, y_true=y_true)
    dice_loss = DiceLoss().call(y_pred=y_pred, y_true=y_true)

    assert is_equal_tf(dice_score, -dice_loss)


def test_separable_filter():
    """Testing separable filter case where diagonal ones are propagated."""
    k = tf.ones(shape=(3, 3, 3, 3, 1), dtype=tf.float32)

    array_eye = np.identity(3)
    x = np.zeros((3, 3, 3, 3, 1))
    x[:, :, 0, 0, 0] = array_eye
    x = tf.convert_to_tensor(x, dtype=tf.float32)

    expected = tf.ones(shape=(3, 3, 3, 3, 1), dtype=tf.float32)
    got = separable_filter(x, k)

    assert is_equal_tf(got, expected)
