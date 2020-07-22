"""
Tests for deepreg/model/layer_util.py in
pytest style
"""
import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.layer_util as layer_util


def check_equal(tensor_1, tensor_2):
    """
    Given two tf tensors return True/False (not tf tensor)
    Tolerate small errors (<1e-6)
    :param tensor_1: Tensor to check equality to against tensor_2.
    :param tensor_2: Tensor to check equality to against tensor_1.
    :return: True if difference less than 1e-6, False otherwise.
    """
    return tf.reduce_max(tf.abs(tensor_1 - tensor_2)).numpy() < 1e-6


def test_check_inputs():
    """
    Test check_inputs by confirming that it accepts proper
    types and handles a few simple cases.
    """
    # Check inputs list - Pass
    assert layer_util.check_inputs([], 0) is None

    # Check inputs tuple - Pass
    assert layer_util.check_inputs((), 0) is None

    # Check inputs int - Fail
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs(0, 0)
    msg = " ".join(execinfo.value.args[0].split())
    assert "Inputs should be a list or tuple" in msg

    # Check inputs float - Fail
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs(0.0, 0)
    msg = " ".join(execinfo.value.args[0].split())
    assert "Inputs should be a list or tuple" in msg

    # Check size float - Fail
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs([1], 0.5)
    msg = " ".join(execinfo.value.args[0].split())
    assert "Inputs should be a list or tuple of size" in msg

    # Check size 0 - Pass
    assert layer_util.check_inputs([], 0) is None
    assert layer_util.check_inputs((), 0) is None

    # Check size 0 - Fail
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs([0], 0)
    msg = " ".join(execinfo.value.args[0].split())
    assert "Inputs should be a list or tuple of size" in msg
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs((0,), 0)
    msg = " ".join(execinfo.value.args[0].split())
    assert "Inputs should be a list or tuple of size" in msg

    # Check size 1 - Pass
    assert layer_util.check_inputs([0], 1) is None
    assert layer_util.check_inputs((0,), 1) is None

    # Check size 1 - Fail
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs([], 1)
    msg = " ".join(execinfo.value.args[0].split())
    assert "Inputs should be a list or tuple of size" in msg
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs((), 1)
    msg = " ".join(execinfo.value.args[0].split())
    assert "Inputs should be a list or tuple of size" in msg

    # Check msg spacing - Pass
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs(0, 0, msg="Start of message")
    msg = " ".join(execinfo.value.args[0].split())
    assert "Start of message" in msg


def test_get_reference_grid():
    """
    Test get_reference_grid by confirming that it generates
    a sample grid test case to check_equal's tolerance level.
    """
    want = tf.constant(
        np.array(
            [[[[0, 0, 0], [0, 0, 1], [0, 0, 2]], [[0, 1, 0], [0, 1, 1], [0, 1, 2]]]],
            dtype=np.float32,
        )
    )
    get = layer_util.get_reference_grid(grid_size=[1, 2, 3])
    assert check_equal(want, get)


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


def test_pyramid_combinations():
    """
    Test pyramid_combinations by confirming that it generates
    appropriate solutions for simple 1D and 2D cases.
    """
    # Check numerical outputs are correct for a simple 1D pair of weights, values - Pass
    weights = tf.constant(np.array([[0.2]], dtype=np.float32))
    values = tf.constant(np.array([[1], [2]], dtype=np.float32))
    # expected = 1 * 0.2 + 2 * 2
    expected = tf.constant(np.array([1.8], dtype=np.float32))
    got = layer_util.pyramid_combination(values=values, weights=weights)
    assert check_equal(got, expected)

    # Check numerical outputs are correct for a 2D pair of weights, values - Pass
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
    got = layer_util.pyramid_combination(values=values, weights=weights)
    assert check_equal(got, expected)

    # Check input lengths match - Fail
    weights = tf.constant(np.array([[[0.2]], [[0.2]]], dtype=np.float32))
    values = tf.constant(np.array([[1], [2]], dtype=np.float32))
    with pytest.raises(ValueError) as execinfo:
        layer_util.pyramid_combination(values=values, weights=weights)
    msg = " ".join(execinfo.value.args[0].split())
    assert (
        "In pyramid_combination, elements of values and weights should have same dimension"
        in msg
    )

    # Check input lengths match - Fail
    weights = tf.constant(np.array([[0.2]], dtype=np.float32))
    values = tf.constant(np.array([[1]], dtype=np.float32))
    with pytest.raises(ValueError) as execinfo:
        layer_util.pyramid_combination(values=values, weights=weights)
    msg = " ".join(execinfo.value.args[0].split())
    assert (
        "In pyramid_combination, num_dim = len(weights), len(values) must be 2 ** num_dim"
        in msg
    )


def test_resample():
    """
    Test resample by confirming that it generates appropriate
    resampling on two test cases with outputs within check_equal's
    tolerance level, and one which should fail (incompatible shapes).
    """
    # linear, vol has no feature channel - Pass
    interpolation = "linear"
    vol = tf.constant(
        np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.float32)
    )  # shape = [1,2,3]
    loc = tf.constant(
        np.array(
            [
                [
                    [[0, 0], [0, 1], [0, 3]],  # outside frame
                    [[0.4, 0], [0.5, 1], [0.6, 2]],
                    [[0.4, 0.7], [0.5, 0.5], [0.6, 0.3]],
                ]
            ],  # resampled = 3x+y
            dtype=np.float32,
        )
    )  # shape = [1,3,3,2]
    want = tf.constant(
        np.array([[[0, 1, 2], [1.2, 2.5, 3.8], [1.9, 2, 2.1]]], dtype=np.float32)
    )  # shape = [1,3,3]
    get = layer_util.resample(vol=vol, loc=loc, interpolation=interpolation)
    assert check_equal(want, get)

    # linear, vol has feature channel - Pass
    interpolation = "linear"
    vol = tf.constant(
        np.array(
            [[[[0, 0], [1, 1], [2, 2]], [[3, 3], [4, 4], [5, 5]]]], dtype=np.float32
        )
    )  # shape = [1,2,3,2]
    loc = tf.constant(
        np.array(
            [
                [
                    [[0, 0], [0, 1], [0, 3]],  # outside frame
                    [[0.4, 0], [0.5, 1], [0.6, 2]],
                    [[0.4, 0.7], [0.5, 0.5], [0.6, 0.3]],
                ]
            ],  # resampled = 3x+y
            dtype=np.float32,
        )
    )  # shape = [1,3,3,2]
    want = tf.constant(
        np.array(
            [
                [
                    [[0, 0], [1, 1], [2, 2]],
                    [[1.2, 1.2], [2.5, 2.5], [3.8, 3.8]],
                    [[1.9, 1.9], [2, 2], [2.1, 2.1]],
                ]
            ],
            dtype=np.float32,
        )
    )  # shape = [1,3,3,2]
    get = layer_util.resample(vol=vol, loc=loc, interpolation=interpolation)
    assert check_equal(want, get)

    # Inconsistent shapes for resampling - Fail
    interpolation = "linear"
    vol = tf.constant(np.array([[0]], dtype=np.float32))  # shape = [1,1]
    loc = tf.constant(np.array([[0, 0], [0, 0]], dtype=np.float32))  # shape = [2,2]
    with pytest.raises(ValueError) as execinfo:
        layer_util.resample(vol=vol, loc=loc, interpolation=interpolation)
    msg = " ".join(execinfo.value.args[0].split())
    assert "vol shape inconsistent with loc" in msg

    # Non-'linear' resampling - Fail
    interpolation = "some-string"
    vol = tf.constant(np.array([[0]], dtype=np.float32))  # shape = [1,1]
    loc = tf.constant(np.array([[0, 0], [0, 0]], dtype=np.float32))  # shape = [2,2]
    with pytest.raises(ValueError) as execinfo:
        layer_util.resample(vol=vol, loc=loc, interpolation=interpolation)
    msg = " ".join(execinfo.value.args[0].split())
    assert "resample supports only linear interpolation" in msg


def test_random_transform_generator():
    """
    Test random_transform_generator by confirming that it generates
    appropriate solutions and output sizes for seeded examples.
    """
    # Check shapes are correct Batch Size = 1 - Pass
    batch_size = 1
    transforms = layer_util.random_transform_generator(batch_size, 0)
    assert transforms.shape == (batch_size, 4, 3)

    # Check numerical outputs are correct for a given seed - Pass
    batch_size = 1
    scale = 0.1
    seed = 0
    expected = tf.constant(
        np.array(
            [
                [
                    [9.4661278e-01, -3.8267835e-03, 3.6934228e-03],
                    [5.5613145e-03, 9.8034811e-01, -1.8044969e-02],
                    [1.9651605e-04, 1.4576728e-02, 9.6243286e-01],
                    [-2.5107686e-03, 1.9579126e-02, -1.2195010e-02],
                ]
            ],
            dtype=np.float32,
        )
    )  # shape = (1, 4, 3)
    got = layer_util.random_transform_generator(
        batch_size=batch_size, scale=scale, seed=seed
    )
    assert check_equal(got, expected)


def test_warp_grid():
    """
    Test warp_grid by confirming that it generates
    appropriate solutions for a simple precomputed case.
    """
    grid = tf.constant(
        np.array(
            [[[[0, 0, 0], [0, 0, 1], [0, 0, 2]], [[0, 1, 0], [0, 1, 1], [0, 1, 2]]]],
            dtype=np.float32,
        )
    )  # shape = (1, 2, 3, 3)
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
    got = layer_util.warp_grid(grid=grid, theta=theta)
    assert check_equal(got, expected)


def test_resize3d():
    """
    Test resize3d by confirming the output shapes.
    """

    # Check resize3d for images with different size and without channel nor batch - Pass
    input_shape = (1, 3, 5)
    output_shape = (2, 4, 6)
    size = (2, 4, 6)
    got = layer_util.resize3d(image=tf.ones(input_shape), size=size)
    assert got.shape == output_shape

    # Check resize3d for images with different size and without channel - Pass
    input_shape = (1, 1, 3, 5)
    output_shape = (1, 2, 4, 6)
    size = (2, 4, 6)
    got = layer_util.resize3d(image=tf.ones(input_shape), size=size)
    assert got.shape == output_shape

    # Check resize3d for images with different size and with one channel - Pass
    input_shape = (1, 1, 3, 5, 1)
    output_shape = (1, 2, 4, 6, 1)
    size = (2, 4, 6)
    got = layer_util.resize3d(image=tf.ones(input_shape), size=size)
    assert got.shape == output_shape

    # Check resize3d for images with different size and with multiple channels - Pass
    input_shape = (1, 1, 3, 5, 3)
    output_shape = (1, 2, 4, 6, 3)
    size = (2, 4, 6)
    got = layer_util.resize3d(image=tf.ones(input_shape), size=size)
    assert got.shape == output_shape

    # Check resize3d for images with the same size and without channel nor batch - Pass
    input_shape = (1, 3, 5)
    output_shape = (1, 3, 5)
    size = (1, 3, 5)
    got = layer_util.resize3d(image=tf.ones(input_shape), size=size)
    assert got.shape == output_shape

    # Check resize3d for images with the same size and without channel - Pass
    input_shape = (1, 1, 3, 5)
    output_shape = (1, 1, 3, 5)
    size = (1, 3, 5)
    got = layer_util.resize3d(image=tf.ones(input_shape), size=size)
    assert got.shape == output_shape

    # Check resize3d for images with the same size and with one channel - Pass
    input_shape = (1, 1, 3, 5, 1)
    output_shape = (1, 1, 3, 5, 1)
    size = (1, 3, 5)
    got = layer_util.resize3d(image=tf.ones(input_shape), size=size)
    assert got.shape == output_shape

    # Check resize3d for images with the same size and with multiple channels - Pass
    input_shape = (1, 1, 3, 5, 3)
    output_shape = (1, 1, 3, 5, 3)
    size = (1, 3, 5)
    got = layer_util.resize3d(image=tf.ones(input_shape), size=size)
    assert got.shape == output_shape

    # Check resize3d for proper image dimensions - Fail
    input_shape = (1, 1)
    size = (1, 1, 1)
    with pytest.raises(ValueError) as execinfo:
        layer_util.resize3d(image=tf.ones(input_shape), size=size)
    msg = " ".join(execinfo.value.args[0].split())
    assert "resize3d takes input image of dimension 3 or 4 or 5" in msg

    # Check resize3d for proper size - Fail
    input_shape = (1, 1, 1)
    size = (1, 1)
    with pytest.raises(ValueError) as execinfo:
        layer_util.resize3d(image=tf.ones(input_shape), size=size)
    msg = " ".join(execinfo.value.args[0].split())
    assert "resize3d takes size of type tuple/list and of length 3" in msg
