"""
Tests for deepreg/model/layer_util.py in
pytest style
"""
from test.unit.util import is_equal_tf
from typing import Tuple, Union

import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.layer_util as layer_util


def test_get_reference_grid():
    """
    Test get_reference_grid by confirming that it generates
    a sample grid test case to is_equal_tf's tolerance level.
    """
    want = tf.constant(
        np.array(
            [[[[0, 0, 0], [0, 0, 1], [0, 0, 2]], [[0, 1, 0], [0, 1, 1], [0, 1, 2]]]],
            dtype=np.float32,
        )
    )
    get = layer_util.get_reference_grid(grid_size=[1, 2, 3])
    assert is_equal_tf(want, get)


def test_get_n_bits_combinations():
    """
    Test get_n_bits_combinations by confirming that it generates
    appropriate solutions for 1D, 2D, and 3D cases.
    """
    # Check n=1 - Pass
    assert layer_util.get_n_bits_combinations(1) == [[0], [1]]
    # Check n=2 - Pass
    assert layer_util.get_n_bits_combinations(2) == [[0, 0], [0, 1], [1, 0], [1, 1]]

    # Check n=3 - Pass
    assert layer_util.get_n_bits_combinations(3) == [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]


class TestPyramidCombination:
    def test_1d(self):
        weights = tf.constant(np.array([[0.2]], dtype=np.float32))
        values = tf.constant(np.array([[1], [2]], dtype=np.float32))

        # expected = 1 * 0.2 + 2 * 2
        expected = tf.constant(np.array([1.8], dtype=np.float32))
        got = layer_util.pyramid_combination(
            values=values, weight_floor=weights, weight_ceil=1 - weights
        )
        assert is_equal_tf(got, expected)

    def test_2d(self):
        weights = tf.constant(np.array([[0.2], [0.3]], dtype=np.float32))
        values = tf.constant(
            np.array(
                [
                    [1],  # value at corner (0, 0), weight = 0.2 * 0.3
                    [2],  # value at corner (0, 1), weight = 0.2 * 0.7
                    [3],  # value at corner (1, 0), weight = 0.8 * 0.3
                    [4],  # value at corner (1, 1), weight = 0.8 * 0.7
                ],
                dtype=np.float32,
            )
        )
        # expected = 1 * 0.2 * 0.3
        #          + 2 * 0.2 * 0.7
        #          + 3 * 0.8 * 0.3
        #          + 4 * 0.8 * 0.7
        expected = tf.constant(np.array([3.3], dtype=np.float32))
        got = layer_util.pyramid_combination(
            values=values, weight_floor=weights, weight_ceil=1 - weights
        )
        assert is_equal_tf(got, expected)

    def test_error_dim(self):
        weights = tf.constant(np.array([[[0.2]], [[0.2]]], dtype=np.float32))
        values = tf.constant(np.array([[1], [2]], dtype=np.float32))
        with pytest.raises(ValueError) as err_info:
            layer_util.pyramid_combination(
                values=values, weight_floor=weights, weight_ceil=1 - weights
            )
        assert (
            "In pyramid_combination, elements of values, weight_floor, "
            "and weight_ceil should have same dimension" in str(err_info.value)
        )

    def test_error_len(self):
        weights = tf.constant(np.array([[0.2]], dtype=np.float32))
        values = tf.constant(np.array([[1]], dtype=np.float32))
        with pytest.raises(ValueError) as err_info:
            layer_util.pyramid_combination(
                values=values, weight_floor=weights, weight_ceil=1 - weights
            )
        assert (
            "In pyramid_combination, num_dim = len(weight_floor), "
            "len(values) must be 2 ** num_dim" in str(err_info.value)
        )


class TestLinearResample:
    x_min, x_max = 0, 2
    y_min, y_max = 0, 2
    # vol are values on grid [0,2]x[0,2]
    # values on each point is 3x+y
    # shape = (1,3,3)
    vol = tf.constant(np.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]]), dtype=tf.float32)
    # loc are some points, especially
    # shape = (1,4,3,2)
    loc = tf.constant(
        np.array(
            [
                [
                    [[0, 0], [0, 1], [1, 2]],  # boundary corners
                    [[0.4, 0], [0.5, 2], [2, 1.7]],  # boundary edge
                    [[-0.4, 0.7], [0, 3], [2, 3]],  # outside boundary
                    [[0.4, 0.7], [1, 1], [0.6, 0.3]],  # internal
                ]
            ]
        ),
        dtype=tf.float32,
    )

    @pytest.mark.parametrize("channel", [0, 1, 2])
    def test_repeat_extrapolation(self, channel):
        x = self.loc[..., 0]
        y = self.loc[..., 1]
        x = tf.clip_by_value(x, self.x_min, self.x_max)
        y = tf.clip_by_value(y, self.y_min, self.y_max)
        expected = 3 * x + y

        vol = self.vol
        if channel > 0:
            vol = tf.repeat(vol[..., None], channel, axis=-1)
            expected = tf.repeat(expected[..., None], channel, axis=-1)

        got = layer_util.resample(vol=vol, loc=self.loc, zero_boundary=False)
        assert is_equal_tf(expected, got)

    @pytest.mark.parametrize("channel", [0, 1, 2])
    def test_repeat_zero_bound(self, channel):
        x = self.loc[..., 0]
        y = self.loc[..., 1]
        expected = 3 * x + y
        expected = (
            expected
            * tf.cast(x > self.x_min, tf.float32)
            * tf.cast(x <= self.x_max, tf.float32)
        )
        expected = (
            expected
            * tf.cast(y > self.y_min, tf.float32)
            * tf.cast(y <= self.y_max, tf.float32)
        )

        vol = self.vol
        if channel > 0:
            vol = tf.repeat(vol[..., None], channel, axis=-1)
            expected = tf.repeat(expected[..., None], channel, axis=-1)

        got = layer_util.resample(vol=vol, loc=self.loc, zero_boundary=True)
        assert is_equal_tf(expected, got)

    def test_shape_error(self):
        vol = tf.constant(np.array([[0]], dtype=np.float32))  # shape = [1,1]
        loc = tf.constant(np.array([[0, 0], [0, 0]], dtype=np.float32))  # shape = [2,2]
        with pytest.raises(ValueError) as err_info:
            layer_util.resample(vol=vol, loc=loc)
        assert "vol shape inconsistent with loc" in str(err_info.value)

    def test_interpolation_error(self):
        interpolation = "nearest"
        vol = tf.constant(np.array([[0]], dtype=np.float32))  # shape = [1,1]
        loc = tf.constant(np.array([[0, 0], [0, 0]], dtype=np.float32))  # shape = [2,2]
        with pytest.raises(ValueError) as err_info:
            layer_util.resample(vol=vol, loc=loc, interpolation=interpolation)
        assert "resample supports only linear interpolation" in str(err_info.value)


class TestWarpGrid:
    """
    Test warp_grid by confirming that it generates
    appropriate solutions for simple precomputed cases.
    """

    grid = tf.constant(
        np.array(
            [[[[0, 0, 0], [0, 0, 1], [0, 0, 2]], [[0, 1, 0], [0, 1, 1], [0, 1, 2]]]],
            dtype=np.float32,
        )
    )  # shape = (1, 2, 3, 3)

    def test_identical(self):
        theta = tf.constant(np.eye(4, 3).reshape((1, 4, 3)), dtype=tf.float32)
        expected = self.grid[None, ...]  # shape = (1, 1, 2, 3, 3)
        got = layer_util.warp_grid(grid=self.grid, theta=theta)
        assert is_equal_tf(got, expected)

    def test_non_identical(self):
        theta = tf.constant(
            np.array(
                [
                    [
                        [0.86, 0.75, 0.48],
                        [0.07, 0.98, 0.01],
                        [0.72, 0.52, 0.97],
                        [0.12, 0.4, 0.04],
                    ]
                ],
                dtype=np.float32,
            )
        )  # shape = (1, 4, 3)
        expected = tf.constant(
            np.array(
                [
                    [
                        [
                            [[0.12, 0.4, 0.04], [0.84, 0.92, 1.01], [1.56, 1.44, 1.98]],
                            [[0.19, 1.38, 0.05], [0.91, 1.9, 1.02], [1.63, 2.42, 1.99]],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
        )  # shape = (1, 1, 2, 3, 3)
        got = layer_util.warp_grid(grid=self.grid, theta=theta)
        assert is_equal_tf(got, expected)


class TestGaussianFilter3D:
    @pytest.mark.parametrize(
        "kernel_sigma, kernel_size",
        [
            ((1, 1, 1), (3, 3, 3, 3, 3)),
            ((2, 2, 2), (7, 7, 7, 3, 3)),
            ((5, 5, 5), (15, 15, 15, 3, 3)),
            (1, (3, 3, 3, 3, 3)),
            (2, (7, 7, 7, 3, 3)),
            (5, (15, 15, 15, 3, 3)),
        ],
    )
    def test_kernel_size(self, kernel_sigma, kernel_size):
        filter = layer_util.gaussian_filter_3d(kernel_sigma)
        assert filter.shape == kernel_size

    @pytest.mark.parametrize(
        "kernel_sigma",
        [(1, 1, 1), (2, 2, 2), (5, 5, 5)],
    )
    def test_sum(self, kernel_sigma):
        filter = layer_util.gaussian_filter_3d(kernel_sigma)
        assert np.allclose(np.sum(filter), 3, atol=1e-3)


class TestDeconvOutputPadding:
    @pytest.mark.parametrize(
        ("input_shape", "output_shape", "kernel_size", "stride", "padding", "expected"),
        [
            (5, 5, 3, 1, "same", 0),
            (5, 7, 3, 1, "valid", 0),
            (5, 3, 3, 1, "full", 0),
            (5, 6, 3, 1, "same", 1),
            (5, 8, 3, 1, "valid", 1),
            (5, 4, 3, 1, "full", 1),
            (5, 9, 3, 2, "same", 0),
            (5, 11, 3, 2, "valid", 0),
            (5, 7, 3, 2, "full", 0),
        ],
    )
    def test_1d(
        self,
        input_shape: int,
        output_shape: int,
        kernel_size: int,
        stride: int,
        padding: str,
        expected: int,
    ):
        """
        Test _deconv_output_padding by verifying output

        :param input_shape: shape of Conv3DTranspose input tensor
        :param output_shape: shape of Conv3DTranspose output tensor
        :param kernel_size: kernel size of Conv3DTranspose layer
        :param stride: stride of Conv3DTranspose layer
        :param padding: padding of Conv3DTranspose layer
        :param expected: expected output padding for Conv3DTranspose layer
        """
        got = layer_util._deconv_output_padding(
            input_shape, output_shape, kernel_size, stride, padding
        )
        assert got == expected

    def test_1d_err(self):
        """Test _deconv_output_padding err raising."""
        with pytest.raises(ValueError) as err_info:
            layer_util._deconv_output_padding(5, 5, 3, 1, "x")
        assert "Unknown padding" in str(err_info.value)

    @pytest.mark.parametrize(
        ("input_shape", "output_shape", "kernel_size", "stride", "padding", "expected"),
        [
            (5, 9, 3, 2, "same", 0),
            ((5, 5), (9, 10), 3, 2, "same", (0, 1)),
            ((5, 5, 6), (9, 10, 12), 3, 2, "same", (0, 1, 1)),
            ((5, 5), (9, 10), (3, 3), 2, "same", (0, 1)),
            ((5, 5), (9, 10), 3, (2, 2), "same", (0, 1)),
            ((5, 5), (9, 10), (3, 4), 2, "same", (0, 2)),
        ],
    )
    def test_n_dim(
        self,
        input_shape: Union[Tuple[int, ...], int],
        output_shape: Union[Tuple[int, ...], int],
        kernel_size: Union[Tuple[int, ...], int],
        stride: Union[Tuple[int, ...], int],
        padding: str,
        expected: Union[Tuple[int, ...], int],
    ):
        """
        Test deconv_output_padding by verifying output

        :param input_shape: shape of Conv3DTranspose input tensor
        :param output_shape: shape of Conv3DTranspose output tensor
        :param kernel_size: kernel size of Conv3DTranspose layer
        :param stride: stride of Conv3DTranspose layer
        :param padding: padding of Conv3DTranspose layer
        :param expected: expected output padding for Conv3DTranspose layer
        """
        got = layer_util.deconv_output_padding(
            input_shape, output_shape, kernel_size, stride, padding
        )
        assert got == expected
