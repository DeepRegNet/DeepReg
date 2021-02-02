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

import deepreg.loss.image as image


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
        got = image.SumSquaredDifference().call(
            y_true,
            y_pred,
        )
        assert is_equal_tf(got, expected)


class TestGlobalMutualInformation:
    @pytest.mark.parametrize(
        "y_true,y_pred,shape,expected",
        [
            (0.6, 0.3, (3, 3, 3, 3), 0.0),
            (0.6, 0.3, (3, 3, 3, 3, 3), 0.0),
            (0.0, 1.0, (3, 3, 3, 3, 3), 0.0),
        ],
    )
    def test_zero_info(self, y_true, y_pred, shape, expected):
        y_true = y_true * np.ones(shape=shape)
        y_pred = y_pred * np.ones(shape=shape)
        expected = expected * np.ones(shape=(shape[0],))
        got = image.GlobalMutualInformation().call(
            y_true,
            y_pred,
        )
        assert is_equal_tf(got, expected)

    def test_get_config(self):
        got = image.GlobalMutualInformation().get_config()
        expected = dict(
            num_bins=23,
            sigma_ratio=0.5,
            reduction=tf.keras.losses.Reduction.SUM,
            name="GlobalMutualInformation",
        )
        assert got == expected


@pytest.mark.parametrize("kernel_size", [3, 5, 7])
@pytest.mark.parametrize("name", ["gaussian", "triangular", "rectangular"])
def test_kernel_fn(kernel_size, name):
    kernel_fn = image.LocalNormalizedCrossCorrelation.kernel_fn_dict[name]
    filters = kernel_fn(kernel_size)
    assert filters.shape == (kernel_size,)


class TestLocalNormalizedCrossCorrelation:
    @pytest.mark.parametrize(
        "y_true,y_pred,shape,kernel_type,expected",
        [
            (0.6, 0.3, (12, 12, 12, 12), "rectangular", 1.0),
            (0.6, 0.3, (12, 12, 12, 12, 1), "rectangular", 1.0),
            (0.0, 1.0, (12, 12, 12, 12, 1), "rectangular", 1.0),
            (0.6, 0.3, (12, 12, 12, 12, 1), "gaussian", 1.0),
            (0.6, 0.3, (12, 12, 12, 12, 1), "triangular", 1.0),
        ],
    )
    def test_zero_info(self, y_true, y_pred, shape, kernel_type, expected):
        y_true = y_true * tf.ones(shape=shape)
        y_pred = y_pred * tf.ones(shape=shape)
        expected = expected * tf.ones(shape=(shape[0],))
        got = image.LocalNormalizedCrossCorrelation(kernel_type=kernel_type).call(
            y_true,
            y_pred,
        )
        assert is_equal_tf(got, expected)

    def test_error(self):
        y = np.ones(shape=(3, 3, 3, 3))
        with pytest.raises(ValueError) as err_info:
            image.LocalNormalizedCrossCorrelation(kernel_type="constant").call(y, y)
        assert "Wrong kernel_type constant for LNCC loss type." in str(err_info.value)

    def test_get_config(self):
        got = image.LocalNormalizedCrossCorrelation().get_config()
        expected = dict(
            kernel_size=9,
            kernel_type="rectangular",
            reduction=tf.keras.losses.Reduction.SUM,
            name="LocalNormalizedCrossCorrelation",
        )
        assert got == expected


class TestGlobalNormalizedCrossCorrelation:
    @pytest.mark.parametrize(
        "y_true,y_pred,shape,expected",
        [
            (0.6, 0.3, (3, 3), 1),
            (0.6, 0.3, (3, 3, 3), 1),
            (0.6, -0.3, (3, 3, 3), 1),
            (0.6, 0.3, (3, 3, 3, 3), 1),
        ],
    )
    def test_output(self, y_true, y_pred, shape, expected):

        y_true = y_true * tf.ones(shape=shape)
        y_pred = y_pred * tf.ones(shape=shape)

        pad_width = tuple([(0, 0)] + [(1, 1)] * (len(shape) - 1))
        y_true = np.pad(y_true, pad_width=pad_width)
        y_pred = np.pad(y_pred, pad_width=pad_width)

        got = image.GlobalNormalizedCrossCorrelation().call(
            y_true,
            y_pred,
        )

        expected = expected * tf.ones(shape=(shape[0],))

        assert is_equal_tf(got, expected)
