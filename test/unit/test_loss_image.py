"""
Tests for deepreg/model/loss/image.py in
pytest style.
Notes: The format of inputs to the function dissimilarity_fn
in image.py should be better converted into tf tensor type beforehand.
"""
from test.unit.util import is_equal_tf

import pytest
import tensorflow as tf

import deepreg.model.loss.image as image


class TestSSD:
    y_true1 = tf.ones((2, 1, 2, 3, 2), dtype=tf.float32)
    y_pred1 = 0.5 * y_true1
    y_true2 = tf.ones((2, 1, 2, 3), dtype=tf.float32)
    y_pred2 = 0.5 * y_true2

    @pytest.mark.parametrize(
        "y_true,y_pred,expected",
        [
            (y_true1, y_pred1, [0.25, 0.25]),
            (y_true1, -y_pred1, [2.25, 2.25]),
            (y_pred1, y_pred1, [0, 0]),
            (y_pred1, -y_pred1, [1, 1]),
            (y_true2, y_pred2, [0.25, 0.25]),
            (y_true2, -y_pred2, [2.25, 2.25]),
            (y_pred2, y_pred2, [0, 0]),
            (y_pred2, -y_pred2, [1, 1]),
        ],
    )
    def test_output(self, y_true, y_pred, expected):
        """
        Testing ssd function (sum of squared differences) by comparing the output to expected.
        """
        loss = image.SumSquaredDistance()
        got = loss.call(
            y_true,
            y_pred,
        )
        assert is_equal_tf(got, expected)


class TestLNCC:
    y_true = tf.ones((2, 1, 2, 3, 2), dtype=tf.float32)
    y_pred = 0.5 * y_true

    @pytest.mark.parametrize(
        "y_true,y_pred,kernel_type,expected",
        [
            (y_true, y_pred, "rectangular", [1, 1]),
            (y_true, y_pred, "triangular", [1, 1]),
            (y_true, y_pred, "gaussian", [1, 1]),
            (y_pred, y_pred, "rectangular", [1, 1]),
            (y_pred, y_pred, "triangular", [1, 1]),
            (y_pred, y_pred, "gaussian", [1, 1]),
        ],
    )
    def test_output(self, y_true, y_pred, kernel_type, expected):
        """
        Testing computed local normalized cross correlation function by comparing the output to expected.
        """
        loss = image.LocalNormalizedCrossCorrelation3D(kernel_type=kernel_type)
        got = loss.call(
            y_true,
            y_pred,
        )
        assert is_equal_tf(got, expected)

    def test_error(self):
        with pytest.raises(ValueError) as err_info:
            loss = image.LocalNormalizedCrossCorrelation3D(kernel_type="constant")
            loss(
                self.y_true,
                self.y_pred,
            )
        assert "Wrong kernel_type for LNCC loss type." in str(err_info.value)


class TestGMI:
    @pytest.mark.parametrize(
        "y_true,y_pred,expected",
        [
            [tf.zeros((2, 1, 2, 3, 2)), tf.zeros((2, 1, 2, 3, 2)), tf.zeros((2,))],
            [tf.ones((2, 1, 2, 3, 2)), tf.ones((2, 1, 2, 3, 2)), tf.zeros((2,))],
        ],
    )
    def test_output(self, y_true, y_pred, expected):
        """
        Testing computed global mutual information between images
        using image.global_mutual_information by comparing to precomputed.
        """
        loss = image.GlobalMutualInformation3D()
        got = loss.call(y_true=y_true, y_pred=y_pred)
        assert is_equal_tf(got, expected, atol=1.0e-6)
