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
        """
        Testing ssd function (sum of squared differences) by comparing the output to expected.
        """
        y_true = y_true * np.ones(shape=shape)
        y_pred = y_pred * np.ones(shape=shape)
        expected = expected * np.ones(shape=(shape[0],))
        got = image.SumSquaredDistance().call(
            y_true,
            y_pred,
        )
        assert is_equal_tf(got, expected)


class TestGlobalMutualInformation3D:
    @pytest.mark.parametrize(
        "y_true,y_pred,shape,expected",
        [
            (0.6, 0.3, (3, 3, 3, 3), 0.0),
            (0.6, 0.3, (3, 3, 3, 3, 3), 0.0),
            (0.0, 1.0, (3, 3, 3, 3, 3), 0.0),
        ],
    )
    def test_zero_info(self, y_true, y_pred, shape, expected):
        """
        Testing ssd function (sum of squared differences) by comparing the output to expected.
        """
        y_true = y_true * np.ones(shape=shape)
        y_pred = y_pred * np.ones(shape=shape)
        expected = expected * np.ones(shape=(shape[0],))
        got = image.GlobalMutualInformation3D().call(
            y_true,
            y_pred,
        )
        assert is_equal_tf(got, expected)

    def test_get_config(self):
        got = image.GlobalMutualInformation3D().get_config()
        expected = dict(
            num_bins=23,
            sigma_ratio=0.5,
            reduction=tf.keras.losses.Reduction.AUTO,
            name="GlobalMutualInformation3D",
        )
        assert got == expected


class TestLocalNormalizedCrossCorrelation3D:
    @pytest.mark.parametrize(
        "y_true,y_pred,shape,kernel_type,expected",
        [
            (0.6, 0.3, (3, 3, 3, 3), "rectangular", 1.0),
            (0.6, 0.3, (3, 3, 3, 3, 3), "rectangular", 1.0),
            (0.0, 1.0, (3, 3, 3, 3, 3), "rectangular", 1.0),
            (0.6, 0.3, (3, 3, 3, 3, 3), "gaussian", 1.0),
            (0.6, 0.3, (3, 3, 3, 3, 3), "triangular", 1.0),
        ],
    )
    def test_zero_info(self, y_true, y_pred, shape, kernel_type, expected):
        """
        Testing ssd function (sum of squared differences) by comparing the output to expected.
        """
        y_true = y_true * np.ones(shape=shape)
        y_pred = y_pred * np.ones(shape=shape)
        expected = expected * np.ones(shape=(shape[0],))
        got = image.LocalNormalizedCrossCorrelation3D(kernel_type=kernel_type).call(
            y_true,
            y_pred,
        )
        assert is_equal_tf(got, expected)

    def test_error(self):
        y = np.ones(shape=(3, 3, 3, 3))
        with pytest.raises(ValueError) as err_info:
            image.LocalNormalizedCrossCorrelation3D(kernel_type="constant").call(y, y)
        assert "Wrong kernel_type for LNCC loss type." in str(err_info.value)

    def test_get_config(self):
        got = image.LocalNormalizedCrossCorrelation3D().get_config()
        expected = dict(
            kernel_size=9,
            kernel_type="rectangular",
            reduction=tf.keras.losses.Reduction.AUTO,
            name="LocalNormalizedCrossCorrelation3D",
        )
        assert got == expected
