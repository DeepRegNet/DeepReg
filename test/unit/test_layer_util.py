import pytest
import numpy as np
import tensorflow as tf

import deepreg.model.layer_util as layer_util

def check_equal(x, y):
    """
    Given two tf tensors return True/False (not tf tensor)
    Tolerate small errors (<1e-6)
    :param x: Tensor to check equality to against y.
    :param y: Tensor to check equality to against x.
    :return: True if difference less than 1e-6, False otherwise.
    """
    return tf.reduce_max(tf.abs(x - y)).numpy() < 1e-6

def test_he_normal():
    """
    Test he_normal by confirming that it creates an initializer
    which populates tf tensors.
    """
    # Create sample 2D and 3D tensors to check he_normal creates them
    def make_variables(k, initializer):
        return (tf.Variable(initializer(shape=[k, k], dtype=tf.float32)),
                tf.Variable(initializer(shape=[k, k, k], dtype=tf.float32)))

    two_dim_tensor, three_dim_tensor = make_variables(3, layer_util.he_normal())
    assert two_dim_tensor.shape == (3, 3) and three_dim_tensor.shape == (3, 3, 3)

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
    msg = ' '.join(execinfo.value.args[0].split())
    assert 'Inputs should be a list or tuple' in msg

    # Check inputs float - Fail
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs(0.0, 0)
    msg = ' '.join(execinfo.value.args[0].split())
    assert 'Inputs should be a list or tuple' in msg

    # Check size float - Fail
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs([1], 0.5)
    msg = ' '.join(execinfo.value.args[0].split())
    assert 'Inputs should be a list or tuple of size' in msg

    # Check size 0 - Pass
    assert layer_util.check_inputs([], 0) is None
    assert layer_util.check_inputs((), 0) is None

    # Check size 0 - Fail
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs([0], 0)
    msg = ' '.join(execinfo.value.args[0].split())
    assert 'Inputs should be a list or tuple of size' in msg
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs((0,), 0)
    msg = ' '.join(execinfo.value.args[0].split())
    assert 'Inputs should be a list or tuple of size' in msg

    # Check size 1 - Pass
    assert layer_util.check_inputs([0], 1) is None
    assert layer_util.check_inputs((0,), 1) is None

    # Check size 1 - Fail
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs([], 1)
    msg = ' '.join(execinfo.value.args[0].split())
    assert 'Inputs should be a list or tuple of size' in msg
    with pytest.raises(ValueError) as execinfo:
        layer_util.check_inputs((), 1)
    msg = ' '.join(execinfo.value.args[0].split())
    assert 'Inputs should be a list or tuple of size' in msg

def test_get_reference_grid():
    """
    Test get_reference_grid by confirming that it generates
    a sample grid test case to check_equal's tolerance level.
    """
    want = tf.constant(np.array([[[[0, 0, 0],
                                   [0, 0, 1],
                                   [0, 0, 2]],
                                  [[0, 1, 0],
                                   [0, 1, 1],
                                   [0, 1, 2]]]],
                                dtype=np.float32))
    get = layer_util.get_reference_grid(grid_size=[1, 2, 3])
    assert check_equal(want, get)

def test_get_n_bits_combinations():
    """
    Test get_n_bits_combinations by confirming that it generates
    appropriate solutions for 0D, 1D, 2D, and 3D cases.
    """
    assert layer_util.get_n_bits_combinations(0) == [[]]
    assert layer_util.get_n_bits_combinations(1) == [[0],
                                                     [1]]
    assert layer_util.get_n_bits_combinations(2) == [[0, 0],
                                                     [0, 1],
                                                     [1, 0],
                                                     [1, 1]]
    assert layer_util.get_n_bits_combinations(3) == [[0, 0, 0],
                                                     [0, 0, 1],
                                                     [0, 1, 0],
                                                     [0, 1, 1],
                                                     [1, 0, 0],
                                                     [1, 0, 1],
                                                     [1, 1, 0],
                                                     [1, 1, 1]]

def test_pyramid_combinations():
    pass

def test_resample():
    """
    Test resample by confirming that it generates appropriate
    resampling on two test cases with outputs within check_equal's
    tolerance level, and one which should fail (incompatible shapes).
    """
    # linear, vol has no feature channel
    interpolation = "linear"
    vol = tf.constant(np.array(
        [[[0, 1, 2],
          [3, 4, 5]]],
        dtype=np.float32))          # shape = [1,2,3]
    loc = tf.constant(np.array(
        [[[[0, 0],
           [0, 1],
           [0, 3]],             # outside frame
          [[0.4, 0],
           [0.5, 1],
           [0.6, 2]],
          [[0.4, 0.7],
           [0.5, 0.5],
           [0.6, 0.3]]]],           # resampled = 3x+y
        dtype=np.float32))          # shape = [1,3,3,2]
    want = tf.constant(np.array(
        [[[0, 1, 2],
          [1.2, 2.5, 3.8],
          [1.9, 2, 2.1]]],
        dtype=np.float32))          # shape = [1,3,3]
    get = layer_util.resample(vol=vol,
                              loc=loc,
                              interpolation=interpolation)
    assert check_equal(want, get)

    # linear, vol has feature channel
    interpolation = "linear"
    vol = tf.constant(np.array(
        [[[[0, 0],
           [1, 1],
           [2, 2]],
          [[3, 3],
           [4, 4],
           [5, 5]]]],
        dtype=np.float32))          # shape = [1,2,3,2]
    loc = tf.constant(np.array(
        [[[[0, 0],
           [0, 1],
           [0, 3]],                 # outside frame
          [[0.4, 0],
           [0.5, 1],
           [0.6, 2]],
          [[0.4, 0.7],
           [0.5, 0.5],
           [0.6, 0.3]]]],           # resampled = 3x+y
        dtype=np.float32))          # shape = [1,3,3,2]
    want = tf.constant(np.array(
        [[[[0, 0],
           [1, 1],
           [2, 2]],
          [[1.2, 1.2],
           [2.5, 2.5],
           [3.8, 3.8]],
          [[1.9, 1.9],
           [2, 2],
           [2.1, 2.1]]]],
        dtype=np.float32))          # shape = [1,3,3,2]
    get = layer_util.resample(vol=vol,
                              loc=loc,
                              interpolation=interpolation)
    assert check_equal(want, get)

    # Inconsistent shapes for resampling
    interpolation = "linear"
    vol = tf.constant(np.array(
        [[0]],
        dtype=np.float32))          # shape = [1,1]
    loc = tf.constant(np.array(
        [[0, 0],
         [0, 0]],
        dtype=np.float32))          # shape = [2,2]
    with pytest.raises(ValueError) as execinfo:
        layer_util.resample(vol=vol,
                            loc=loc,
                            interpolation=interpolation)
    msg = ' '.join(execinfo.value.args[0].split())
    assert 'vol shape inconsistent with loc' in msg

def test_random_transform_generator():
    pass
    
def test_warp_grid():
    pass
