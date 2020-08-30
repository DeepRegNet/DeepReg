# coding=utf-8

"""
Tests for deepreg/model/loss/label.py in
pytest style
"""
from test.unit.util import is_equal_tf
from types import FunctionType

import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.loss.label as label


def test_gauss_kernel1d_0():
    """
    Testing case where sigma = 0, expect 0 return
    """
    sigma = tf.constant(0, dtype=tf.float32)
    expect = tf.constant(0, dtype=tf.float32)
    get = label.gauss_kernel1d(sigma)
    assert get == expect


def test_gauss_kernel1d_else():
    """
    Testing case where sigma is not 0,
    expect a tensor returned.
    """
    sigma = 3
    get = tf.cast(label.gauss_kernel1d(sigma), dtype=tf.float32)
    expect = [
        np.exp(-0.5 * x ** 2 / sigma ** 2) for x in range(-sigma * 3, sigma * 3 + 1)
    ]
    expect = tf.convert_to_tensor(expect, dtype=tf.float32)
    expect = expect / tf.reduce_sum(expect)
    assert is_equal_tf(get, expect)


def test_cauchy_kernel_0():
    """
    Test case where sigma = 0, expect 0 return.
    """
    sigma = tf.constant(0, dtype=tf.float32)
    expect = tf.constant(0, dtype=tf.float32)
    get = label.cauchy_kernel1d(sigma)
    assert get == expect


def test_cauchy_kernel_else():
    """
    Test case where sigma is not 0, expect
    tensor returned.
    """
    sigma = 3
    get = tf.cast(label.cauchy_kernel1d(sigma), dtype=tf.float32)
    expect = [1 / ((x / sigma) ** 2 + 1) for x in range(-sigma * 5, sigma * 5 + 1)]
    expect = tf.convert_to_tensor(expect, dtype=tf.float32)
    expect = expect / tf.reduce_sum(expect)
    assert is_equal_tf(get, expect)


def test_foreground_prop_binary():
    """
    Test foreground function with a
    tensor of zeros with some ones, asserting
    equal to known precomputed tensor.
    Testing with binary case.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    expect = tf.convert_to_tensor([1.0 / 3, 1.0 / 3, 1.0 / 3], dtype=tf.float32)
    get = label.foreground_proportion(tensor_eye)
    assert is_equal_tf(get, expect)


def test_foreground_prop_simple():
    """
    Test foreground functions with a tensor
    of zeros with some ones and some values below
    one to assert the thresholding works.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, 0, :, :] = 0.4 * array_eye  # 0
    tensor_eye[:, 1, :, :] = array_eye
    tensor_eye[:, 2, :, :] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)
    expect = [54 / (27 * 9), 54 / (27 * 9), 54 / (27 * 9)]
    get = label.foreground_proportion(tensor_eye)
    assert is_equal_tf(get, expect)


def test_jaccard_index():
    """
    Testing jaccard index function with computed
    tensor.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, 0:2, :, :] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    num = np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6]) - num

    get = num / denom
    expect = label.jaccard_index(tensor_eye, tensor_pred)
    assert is_equal_tf(get, expect)


def test_dice_not_binary():
    """
    Testing dice score with binary tensor
    comparing to a precomputed value.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, 0:2, :, :] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    num = 2 * np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6])

    get = num / denom
    expect = label.dice_score(tensor_eye, tensor_pred)
    assert is_equal_tf(get, expect)


def test_dice_binary():
    """
    Testing dice score with not binary tensor
    to assert thresholding works.
    """
    array_eye = 0.6 * np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, 0:2, :, :] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    num = 2 * np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6])

    get = num / denom
    expect = label.dice_score(tensor_eye, tensor_pred, binary=True)
    assert is_equal_tf(get, expect)


def test_dice_general():
    """
    Testing general dice function with
    non binary features and checking
    against precomputed tensor.
    """
    array_eye = 0.6 * np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, 0:2, :, :] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    y_prod = np.sum(tensor_eye * tensor_pred, axis=(1, 2, 3))
    y_sum = np.sum(tensor_eye, axis=(1, 2, 3)) + np.sum(tensor_pred, axis=(1, 2, 3))

    num = 2 * y_prod
    den = y_sum
    expect = num / den
    get = label.dice_score_generalized(tensor_eye, tensor_pred)

    assert is_equal_tf(get, expect)


def test_weighted_bce():
    """
    Checking binary cross entropy calculation
    against a precomputed tensor.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, 0:2, :, :] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    expect = [1.535057, 1.535057, 1.535057]
    get = label.weighted_binary_cross_entropy(tensor_eye, tensor_pred)
    assert is_equal_tf(get, expect)


def test_separable_filter_0():
    """
    Testing separable filter with case where
    0 length vector is passed.
    """
    pass
    # kernel = np.empty((0))
    # array_eye = np.identity(3, dtype=np.float32)
    # get = label.separable_filter3d(array_eye, kernel)
    # expect = array_eye
    # assert is_equal_tf(get, expect)


def test_separable_filter_else():
    """
    Testing separable filter case where non
    zero length tensor is passed to the
    function.
    """
    k = np.ones((3, 3, 3, 3), dtype=np.float32)
    array_eye = np.identity(3, dtype=np.float32)
    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, :, 0, 0] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)
    k = tf.convert_to_tensor(k, dtype=tf.float32)

    expect = np.ones((3, 3, 3, 3), dtype=np.float32)
    expect = tf.convert_to_tensor(expect, dtype=tf.float32)

    get = label.separable_filter3d(tensor_pred, k)
    assert is_equal_tf(get, expect)


def test_compute_centroid():
    """
    Testing compute centroid function
    and comparing to expected values.
    """
    array_ones = np.ones((2, 2))
    tensor_mask = np.zeros((3, 2, 2, 2))
    tensor_mask[0, :, :, :] = array_ones
    tensor_mask = tf.convert_to_tensor(tensor_mask, dtype=tf.float32)

    tensor_grid = np.zeros((2, 2, 2, 3))
    tensor_grid[:, :, :, 0] = array_ones
    tensor_grid = tf.convert_to_tensor(tensor_grid, dtype=tf.float32)

    expect = np.ones((3, 3))
    expect[0, 1:3] = 0
    get = label.compute_centroid(tensor_mask, tensor_grid)
    assert is_equal_tf(get, expect)


def test_compute_centroid_d():
    """
    Testing compute centroid distance between equal
    tensors returns 0s.
    """
    array_ones = np.ones((2, 2))
    tensor_mask = np.zeros((3, 2, 2, 2))
    tensor_mask[0, :, :, :] = array_ones
    tensor_mask = tf.convert_to_tensor(tensor_mask, dtype=tf.float32)

    tensor_grid = np.zeros((2, 2, 2, 3))
    tensor_grid[:, :, :, 0] = array_ones
    tensor_grid = tf.convert_to_tensor(tensor_grid, dtype=tf.float32)

    get = label.compute_centroid_distance(tensor_mask, tensor_mask, tensor_grid)
    expect = np.zeros((3))
    assert is_equal_tf(get, expect)


def test_squared_error():
    """
    Testing squared error function by comparing
    to precomputed tensor.
    """
    tensor_mask = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_mask[0, 0, 0, 0] = 1
    tensor_mask = tf.convert_to_tensor(tensor_mask, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, :, :, :] = 1
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    expect = np.array([26 / 27, 1.0, 1.0])
    get = label.squared_error(tensor_mask, tensor_pred)
    assert is_equal_tf(get, expect)


def test_single_scale_loss_dice():
    """
    Testing single sclare loss returns
    precomputed, known dice loss for given
    inputs.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, 0:2, :, :] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    num = 2 * np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6])

    expect = 1 - (num / denom)
    get = label.single_scale_loss(tensor_eye, tensor_pred, "dice")
    assert is_equal_tf(get, expect)


def test_single_scale_loss_bce():
    """
    Testing bce single scale loss entry
    returns known loss tensor for given inputs.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, 0:2, :, :] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    expect = [1.535057, 1.535057, 1.535057]
    get = label.single_scale_loss(tensor_eye, tensor_pred, "cross-entropy")

    assert is_equal_tf(get, expect)


def test_single_scale_loss_dg():
    """
    Testing generalised dice loss single
    scale loss function returns known loss
    tensor for given inputs.
    """
    array_eye = 0.6 * np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, 0:2, :, :] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    y_prod = np.sum(tensor_eye * tensor_pred, axis=(1, 2, 3))
    y_sum = np.sum(tensor_eye, axis=(1, 2, 3)) + np.sum(tensor_pred, axis=(1, 2, 3))

    num = 2 * y_prod
    den = y_sum
    expect = 1 - num / den
    get = label.single_scale_loss(tensor_eye, tensor_pred, "dice_generalized")
    assert is_equal_tf(get, expect)


def test_single_scale_loss_jacc():
    """
    Testing single scale loss returns known loss
    tensor when called with jaccard argment.
    """
    array_eye = np.identity(3, dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)

    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_pred[:, 0:2, :, :] = array_eye
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    num = np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6]) - num

    expect = 1 - (num / denom)
    get = label.single_scale_loss(tensor_eye, tensor_pred, "jaccard")
    assert is_equal_tf(get, expect)


def test_single_scale_loss_mean_sq():
    """
    Test single scale loss function returns
    known mean sq value tensor when passed with
    mean squared arg,
    """
    tensor_mask = np.zeros((3, 3, 3, 3))
    tensor_mask[0, 0, 0, 0] = 1
    tensor_mask = tf.convert_to_tensor(tensor_mask, dtype=tf.float32)

    tensor_pred = tf.convert_to_tensor(np.ones((3, 3, 3, 3)), dtype=tf.float32)
    expect = tf.convert_to_tensor(np.array([26 / 27, 1.0, 1.0]), dtype=tf.float32)

    get = label.single_scale_loss(tensor_mask, tensor_pred, "mean-squared")
    assert is_equal_tf(get, expect)


def test_single_scale_loss_other():
    """
    Test value error raised if non supported
    string passed to the single scale loss function.
    """
    tensor_eye = tf.convert_to_tensor(np.zeros((3, 3, 3, 3)), dtype=tf.float32)
    tensor_pred = tf.convert_to_tensor(np.zeros((3, 3, 3, 3)), dtype=tf.float32)

    with pytest.raises(ValueError):
        label.single_scale_loss(tensor_eye, tensor_pred, "random")


def test_multi_scale_loss_pred_len():
    """
    Test assertion error raised if a wrongly sized tensor
    is passed to the multi-scale loss function.
    """
    tensor_true = tf.convert_to_tensor(np.zeros((3, 3, 3, 3)), dtype=tf.float32)
    tensor_pred = tf.convert_to_tensor(np.zeros((3, 3, 3)), dtype=tf.float32)
    with pytest.raises(AssertionError):
        label.multi_scale_loss(
            tensor_true, tensor_pred, loss_type="jaccard", loss_scales=[0, 1, 2]
        )


def test_multi_scale_loss_true_len():
    """
    Test assertion error raised if a wrongly sized tensor
    is passed to the multi-scale loss function.
    """
    tensor_true = tf.convert_to_tensor(np.zeros((3, 3, 3)), dtype=tf.float32)
    tensor_pred = tf.convert_to_tensor(np.zeros((3, 3, 3, 3)), dtype=tf.float32)
    with pytest.raises(AssertionError):
        label.multi_scale_loss(
            tensor_true, tensor_pred, loss_type="jaccard", loss_scales=[0, 1, 2]
        )


def test_multi_scale_loss_kernel():
    """
    Test multi-scale loss kernel returns the appropriate
    loss tensor for same inputs and jaccard cal.
    """
    loss_values = [1, 2, 3]
    array_eye = np.identity(3, dtype=np.float32)
    tensor_pred = np.zeros((3, 3, 3, 3), dtype=np.float32)
    tensor_eye = np.zeros((3, 3, 3, 3), dtype=np.float32)

    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_pred[:, :, 0, 0] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.float32)
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)
    expect = tf.constant([0.9938445, 0.9924956, 0.9938445], dtype=tf.float32)
    get = label.multi_scale_loss(tensor_eye, tensor_pred, "jaccard", loss_values)
    assert is_equal_tf(get, expect)


def test_similarity_fn_unknown_loss():
    """
    Test dissimilarity function raises an error
    if an unknonw loss type is passed.
    """
    config = {"name": "random"}
    with pytest.raises(ValueError):
        label.get_dissimilarity_fn(config)


def test_similarity_fn_multi_scale():
    """
    Asserting loss function returned by get dissimilarity
    function when appropriate strings passed.
    """
    config = {"name": "multi_scale", "multi_scale": "jaccard"}
    assert isinstance(label.get_dissimilarity_fn(config), FunctionType)


def test_similarity_fn_single_scale():
    """
    Asserting loss function returned by get dissimilarity
    function when appropriate strings passed.
    """
    config = {"name": "single_scale", "single_scale": "jaccard"}
    assert isinstance(label.get_dissimilarity_fn(config), FunctionType)
