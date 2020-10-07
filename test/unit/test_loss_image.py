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

    def test_lncc(self):
        got = image.dissimilarity_fn(self.y_true, self.y_pred, "lncc")
        expected = [-0.68002254, -0.9608879]
        assert is_equal_tf(got, expected)

        got = image.dissimilarity_fn(self.y_pred, self.y_pred, "lncc")
        expected = [-1, -1]
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
        with pytest.raises(AssertionError):
            image.dissimilarity_fn(self.y_true, self.y_pred, "")


def test_local_normalized_cross_correlation():
    """
    Testing computed local normalized cross correlation function between images using image.local_normalized_cross_correlation by comparing to precomputed.
    """
    tensor_true = np.ones((2, 1, 2, 3, 2))
    tensor_pred = 0.5 * np.ones((2, 1, 2, 3, 2))
    expect = [1, 1]
    get = image.local_normalized_cross_correlation(
        tensor_true,
        tensor_pred,
    )
    assert is_equal_tf(get, expect)


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
