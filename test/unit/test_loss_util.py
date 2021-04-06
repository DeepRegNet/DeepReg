# coding=utf-8

"""
Tests for deepreg/model/loss/label.py in
pytest style
"""

from test.unit.util import is_equal_tf

import numpy as np
import pytest
import tensorflow as tf

from deepreg.loss.util import MultiScaleMixin, NegativeLossMixin, separable_filter


class TestMultiScaleLoss:
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

    get = separable_filter(tensor_pred, k)
    assert is_equal_tf(get, expect)


class MinusClass(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.name = "MinusClass"

    def call(self, y_true, y_pred):
        return y_true - y_pred


class MinusClassLoss(NegativeLossMixin, MinusClass):
    pass


@pytest.mark.parametrize("y_true,y_pred,expected", [(1, 2, 1), (2, 1, -1), (0, 0, 0)])
def test_negative_loss_mixin(y_true, y_pred, expected):
    """
    Testing NegativeLossMixin class that
    inverts the sign of any value
    returned by a function

    :param y_true: int
    :param y_pred: int
    :param expected: int
    :return:
    """

    y_true = tf.constant(y_true, dtype=tf.float32)
    y_pred = tf.constant(y_pred, dtype=tf.float32)

    got = MinusClassLoss().call(
        y_true,
        y_pred,
    )
    assert is_equal_tf(got, expected)
