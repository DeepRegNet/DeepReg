"""
Tests for deepreg/model/loss/image.py in
pytest style.
Notes: The format of inputs to the function dissimilarity_fn
in image.py should be better converted into tf tensor type beforehand.
"""

from test.unit.util import is_equal_tf
from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

import deepreg.loss.image as image
from deepreg.constant import EPS


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
            reduction=tf.keras.losses.Reduction.AUTO,
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
        ("y_true_shape", "y_pred_shape"),
        [
            ((2, 3, 4, 5), (2, 3, 4, 5)),
            ((2, 3, 4, 5), (2, 3, 4, 5, 1)),
            ((2, 3, 4, 5, 1), (2, 3, 4, 5)),
            ((2, 3, 4, 5, 1), (2, 3, 4, 5, 1)),
        ],
    )
    def test_input_shape(self, y_true_shape: Tuple, y_pred_shape: Tuple):
        """
        Test input with / without channel axis.

        :param y_true_shape: input shape for y_true.
        :param y_pred_shape: input shape for y_pred.
        """
        y_true = tf.ones(shape=y_true_shape)
        y_pred = tf.ones(shape=y_pred_shape)
        got = image.LocalNormalizedCrossCorrelation().call(
            y_true,
            y_pred,
        )
        assert got.shape == y_true_shape[:1]

    @pytest.mark.parametrize(
        ("y_true_shape", "y_pred_shape", "name"),
        [
            ((2, 3, 4, 5), (2, 3, 4, 5, 6), "y_pred"),
            ((2, 3, 4, 5, 6), (2, 3, 4, 5), "y_true"),
        ],
    )
    def test_input_shape_err(self, y_true_shape: Tuple, y_pred_shape: Tuple, name: str):
        """
        Current LNCC does not support image having channel dimension > 1.

        :param y_true_shape: input shape for y_true.
        :param y_pred_shape: input shape for y_pred.
        :param name: name of the tensor having error.
        """
        y_true = tf.ones(shape=y_true_shape)
        y_pred = tf.ones(shape=y_pred_shape)
        with pytest.raises(ValueError) as err_info:
            image.LocalNormalizedCrossCorrelation().call(y_true, y_pred)
        assert f"Last dimension of {name} is not one." in str(err_info.value)

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize(
        ("smooth_nr", "smooth_dr", "expected"),
        [
            (1e-5, 1e-5, 1),
            (0, 1e-5, 0),
            (1e-5, 0, np.inf),
            (0, 0, np.nan),
            (1e-7, 1e-7, 1),
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
        Test values in extreme cases where variances are all zero.

        :param value: value for input.
        :param smooth_nr: constant for numerator.
        :param smooth_dr: constant for denominator.
        :param expected: target value.
        """
        kernel_size = 5
        mid = kernel_size // 2
        shape = (1, kernel_size, kernel_size, kernel_size, 1)
        y_true = tf.ones(shape=shape) * value
        y_pred = tf.ones(shape=shape) * value

        got = image.LocalNormalizedCrossCorrelation(
            kernel_size=kernel_size,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
        ).calc_ncc(
            y_true,
            y_pred,
        )
        got = got[0, mid, mid, mid, 0]
        expected = tf.constant(expected)
        assert is_equal_tf(got, expected)

    @pytest.mark.parametrize(
        "kernel_type",
        ["rectangular", "gaussian", "triangular"],
    )
    @pytest.mark.parametrize(
        "kernel_size",
        [3, 5, 7],
    )
    def test_exact_value(self, kernel_type, kernel_size):
        """
        Test the exact value at the center of a cube.

        :param kernel_type: name of kernel.
        :param kernel_size: size of the kernel and the cube.
        """
        # init
        mid = kernel_size // 2
        tf.random.set_seed(0)
        y_true = tf.random.uniform(shape=(1, kernel_size, kernel_size, kernel_size, 1))
        y_pred = tf.random.uniform(shape=(1, kernel_size, kernel_size, kernel_size, 1))
        loss = image.LocalNormalizedCrossCorrelation(
            kernel_type=kernel_type, kernel_size=kernel_size
        )

        # obtained value
        got = loss.calc_ncc(y_true=y_true, y_pred=y_pred)
        got = got[0, mid, mid, mid, 0]  # center voxel

        # target value
        kernel_3d = (
            loss.kernel[:, None, None]
            * loss.kernel[None, :, None]
            * loss.kernel[None, None, :]
        )
        kernel_3d = kernel_3d[None, :, :, :, None]

        y_true_mean = tf.reduce_sum(y_true * kernel_3d) / loss.kernel_vol
        y_true_normalized = y_true - y_true_mean
        y_true_var = tf.reduce_sum(y_true_normalized ** 2 * kernel_3d)

        y_pred_mean = tf.reduce_sum(y_pred * kernel_3d) / loss.kernel_vol
        y_pred_normalized = y_pred - y_pred_mean
        y_pred_var = tf.reduce_sum(y_pred_normalized ** 2 * kernel_3d)

        cross = tf.reduce_sum(y_true_normalized * y_pred_normalized * kernel_3d)
        expected = (cross ** 2 + EPS) / (y_pred_var * y_true_var + EPS)

        # check
        assert is_equal_tf(got, expected)

    def test_kernel_error(self):
        """Test the error message when using wrong kernel."""
        with pytest.raises(ValueError) as err_info:
            image.LocalNormalizedCrossCorrelation(kernel_type="constant")
        assert "Wrong kernel_type constant for LNCC loss type." in str(err_info.value)

    def test_get_config(self):
        """Test the config is saved correctly."""
        got = image.LocalNormalizedCrossCorrelation().get_config()
        expected = dict(
            kernel_size=9,
            kernel_type="rectangular",
            reduction=tf.keras.losses.Reduction.AUTO,
            name="LocalNormalizedCrossCorrelation",
            smooth_nr=1e-5,
            smooth_dr=1e-5,
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
