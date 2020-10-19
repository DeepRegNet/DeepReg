"""
Tests for deepreg/model/loss/image.py in
pytest style.
Notes: The format of inputs to the function dissimilarity_fn
in image.py should be better converted into tf tensor type beforehand.
"""
from test.unit.util import is_equal_tf

import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.loss.image as image


class TestDissimilarityFn:
    """
    Testing computed dissimilarity function by comparing to precomputed,
    the dissimilarity function can be either normalized cross correlation or sum square error function.
    """

    y_true = tf.constant(np.array(range(12)).reshape((2, 1, 2, 3)), dtype=tf.float32)
    y_pred = 0.6 * tf.ones((2, 1, 2, 3), dtype=tf.float32)

    @pytest.mark.parametrize(
        "y_true,y_pred,name,expected",
        [
            (y_true, y_pred, "lncc", [-0.68002254, -0.9608879]),
            (y_pred, y_pred, "lncc", [-1.0, -1.0]),
            (y_pred, 0 * y_pred, "ssd", [0.36, 0.36]),
            (y_pred, y_pred, "ssd", [0.0, 0.0]),
            (y_pred, y_pred, "gmi", [0.0, 0.0]),
        ],
    )
    def test_output(self, y_true, y_pred, name, expected):
        got = image.dissimilarity_fn(y_true, y_pred, name)
        assert is_equal_tf(got, expected)

    def test_error(self):
        # unknown func name
        with pytest.raises(ValueError) as err_info:
            image.dissimilarity_fn(self.y_true, self.y_pred, "")
        assert "Unknown loss type" in str(err_info.value)


class TestSSD:
    y_true = tf.ones((2, 1, 2, 3, 2), dtype=tf.float32)
    y_pred = 0.5 * y_true

    @pytest.mark.parametrize(
        "y_true,y_pred,expected",
        [
            (y_true, y_pred, [0.25, 0.25]),
            (y_true, -y_pred, [2.25, 2.25]),
            (y_pred, y_pred, [0, 0]),
            (y_pred, -y_pred, [1, 1]),
        ],
    )
    def test_output(self, y_true, y_pred, expected):
        """
        Testing ssd function (sum of squared differences) by comparing the output to expected.
        """
        got = image.ssd(
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
        got = image.local_normalized_cross_correlation(
            y_true, y_pred, kernel_type=kernel_type
        )
        assert is_equal_tf(got, expected)

    def test_error(self):
        with pytest.raises(ValueError) as err_info:
            image.local_normalized_cross_correlation(
                self.y_true, self.y_pred, kernel_type="constant"
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
        got = image.global_mutual_information(y_true=y_true, y_pred=y_pred)
        assert is_equal_tf(got, expected, atol=1.0e-6)
