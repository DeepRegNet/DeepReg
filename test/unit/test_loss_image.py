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
        ],
    )
    def test_lncc(self, y_true, y_pred, name, expected):
        got = image.dissimilarity_fn(y_true, y_pred, name)
        assert is_equal_tf(got, expected)

    def test_ssd(self):
        got = image.dissimilarity_fn(self.y_pred * 0, self.y_pred, "ssd")
        expected = [0.36, 0.36]
        assert is_equal_tf(got, expected)

        got = image.dissimilarity_fn(self.y_pred, self.y_pred, "ssd")
        expected = [0, 0]
        assert is_equal_tf(got, expected)

    def test_gmi(self):
        # TODO gmi diff images
        # gmi same image
        t = tf.ones([4, 3, 3, 3])
        get_zero_similarity_gmi = image.dissimilarity_fn(t, t, "gmi")
        assert is_equal_tf(get_zero_similarity_gmi, [0, 0, 0, 0], atol=1.0e-6)

    def test_error(self):
        # unknown func name
        with pytest.raises(ValueError) as err_info:
            image.dissimilarity_fn(self.y_true, self.y_pred, "")
        assert "Unknown loss type" in str(err_info.value)


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


def test_ssd():
    """
    Testing computed sum squared error function between images using image.ssd by comparing to precomputed.
    """
    tensor_true = 0.3 * np.array(range(108)).reshape((2, 3, 3, 3, 2))
    tensor_pred = 0.1 * np.ones((2, 3, 3, 3, 2))
    tensor_pred[:, :, :, :, :] = 1
    get = image.ssd(tensor_true, tensor_pred)
    expect = [70.165, 557.785]
    assert is_equal_tf(get, expect)


@pytest.mark.parametrize(
    "y_true,y_pred,expected",
    [
        [tf.zeros((4, 3, 3, 3, 1)), tf.zeros((4, 3, 3, 3, 1)), tf.zeros((4,))],
        [tf.ones((4, 3, 3, 3, 1)), tf.ones((4, 3, 3, 3, 1)), tf.zeros((4,))],
    ],
)
def test_gmi(y_true, y_pred, expected):
    """
    Testing computed global mutual information between images
    using image.global_mutual_information by comparing to precomputed.
    TODO when y_true == y_pred, the return is not exactly zero
    """
    got = image.global_mutual_information(y_true=y_true, y_pred=y_pred)
    assert is_equal_tf(got, expected, atol=1.0e-6)
