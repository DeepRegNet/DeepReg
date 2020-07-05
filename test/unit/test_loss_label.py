# coding=utf-8

"""
Tests for deepreg/model/loss/label.py in
pytest style
"""
from types import FunctionType

import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.loss.label as label


def assertTensorsEqual(x, y):
    """
    given two tf tensors return True/False (not tf tensor)
    tolerate small errors
    :param x:
    :param y:
    :return:
    """
    return tf.reduce_max(tf.abs(x - y)).numpy() < 1e-6


def test_gauss_kernel1d_0():
    sigma = 0
    get = label.gauss_kernel1d(sigma)
    expect = tf.constant(0)
    assert get == expect


def test_gauss_kernel1d_else():
    sigma = 3
    get = tf.cast(label.gauss_kernel1d(sigma), dtype=tf.float64)
    list_vals = range(-sigma * 3, sigma * 3 + 1)
    exp = [np.exp(-0.5 * x ** 2 / sigma ** 2) for x in list_vals]
    expect = tf.convert_to_tensor(exp, dtype=tf.float64)
    expect = expect / tf.reduce_sum(expect)
    assert assertTensorsEqual(get, expect)


def test_cauchy_kernel_0():
    sigma = 0
    get = label.cauchy_kernel1d(sigma)
    expect = 0
    assert get == expect


def test_cauchy_kernel_else():
    sigma = 3
    get = tf.cast(label.cauchy_kernel1d(sigma), dtype=tf.float64)
    list_vals = range(-sigma * 5, sigma * 5 + 1)
    exp = [1 / ((x / sigma) ** 2 + 1) for x in list_vals]
    expect = tf.convert_to_tensor(exp, dtype=tf.float64)
    expect = expect / tf.reduce_sum(expect)
    assert assertTensorsEqual(get, expect)


def test_foreground_prop_binary():
    array_eye = np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye
    expect = [1.0 / 3, 1.0 / 3, 1.0 / 3]
    get = label.foreground_proportion(tensor_eye)
    assert assertTensorsEqual(get, expect)


def test_foreground_prop_simple():
    array_eye = np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, 0, :, :] = 0.4 * array_eye  #  0
    tensor_eye[:, 1, :, :] = array_eye
    tensor_eye[:, 2, :, :] = array_eye
    expect = [54 / (27 * 9), 54 / (27 * 9), 54 / (27 * 9)]
    get = label.foreground_proportion(tensor_eye)
    assert assertTensorsEqual(get, expect)


def test_jaccard_index():
    array_eye = np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, 0:2, :, :] = array_eye
    num = np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6]) - num

    get = num / denom
    expect = label.jaccard_index(tensor_eye, tensor_pred)
    assert assertTensorsEqual(get, expect)


def test_dice_not_binary():
    array_eye = np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, 0:2, :, :] = array_eye

    num = 2 * np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6])

    get = num / denom
    expect = label.dice_score(tensor_eye, tensor_pred)
    assert assertTensorsEqual(get, expect)


def test_dice_binary():
    array_eye = 0.6 * np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, 0:2, :, :] = array_eye

    num = 2 * np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6])

    get = num / denom
    expect = label.dice_score(tensor_eye, tensor_pred, binary=True)
    assert assertTensorsEqual(get, expect)


def test_dice_general():
    array_eye = 0.6 * np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, 0:2, :, :] = array_eye

    y_prod = np.sum(tensor_eye * tensor_pred, axis=(1, 2, 3))
    y_sum = np.sum(tensor_eye, axis=(1, 2, 3)) + np.sum(tensor_pred, axis=(1, 2, 3))

    num = 2 * y_prod
    den = y_sum
    expect = num / den
    get = label.dice_score_generalized(tensor_eye, tensor_pred)

    assert assertTensorsEqual(get, expect)


def test_weighted_bce():
    array_eye = np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, 0:2, :, :] = array_eye

    expect = [1.535057, 1.535057, 1.535057]
    get = label.weighted_binary_cross_entropy(tensor_eye, tensor_pred)
    assert assertTensorsEqual(get, expect)


def test_separable_filter_0():
    pass
    # kernel = np.empty((0))
    # array_eye = np.identity((3))
    # get = label.separable_filter3d(array_eye, kernel)
    # expect = array_eye
    # assert assertTensorsEqual(get, expect)


def test_separable_filter_else():
    k = np.ones((3, 3, 3, 3))
    array_eye = np.identity((3))
    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, :, 0, 0] = array_eye

    expect = np.ones((3, 3, 3, 3))

    get = label.separable_filter3d(tensor_pred, k)
    assert assertTensorsEqual(get, expect)


def test_compute_centroid():

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
    assert assertTensorsEqual(get, expect)


def test_compute_centroid_d():
    array_ones = np.ones((2, 2))
    tensor_mask = np.zeros((3, 2, 2, 2))
    tensor_mask[0, :, :, :] = array_ones
    tensor_mask = tf.convert_to_tensor(tensor_mask, dtype=tf.float32)

    tensor_grid = np.zeros((2, 2, 2, 3))
    tensor_grid[:, :, :, 0] = array_ones
    tensor_grid = tf.convert_to_tensor(tensor_grid, dtype=tf.float32)

    get = label.compute_centroid_distance(tensor_mask, tensor_mask, tensor_grid)
    expect = np.zeros((3))
    assert assertTensorsEqual(get, expect)


def test_squared_error():
    tensor_mask = np.zeros((3, 3, 3, 3))
    tensor_mask[0, 0, 0, 0] = 1

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, :, :, :] = 1
    expect = np.array([26 / 27, 1.0, 1.0])
    get = label.squared_error(tensor_mask, tensor_pred)
    assert assertTensorsEqual(get, expect)


def test_single_scale_loss_dice():
    array_eye = np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, 0:2, :, :] = array_eye

    num = 2 * np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6])

    expect = 1 - (num / denom)
    get = label.single_scale_loss(tensor_eye, tensor_pred, "dice")
    assert assertTensorsEqual(get, expect)


def test_single_scale_loss_bce():
    array_eye = np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, 0:2, :, :] = array_eye

    expect = [1.535057, 1.535057, 1.535057]
    get = label.single_scale_loss(tensor_eye, tensor_pred, "cross-entropy")

    assert assertTensorsEqual(get, expect)


def test_single_scale_loss_dg():
    array_eye = 0.6 * np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, 0:2, :, :] = array_eye

    y_prod = np.sum(tensor_eye * tensor_pred, axis=(1, 2, 3))
    y_sum = np.sum(tensor_eye, axis=(1, 2, 3)) + np.sum(tensor_pred, axis=(1, 2, 3))

    num = 2 * y_prod
    den = y_sum
    expect = 1 - num / den
    get = label.single_scale_loss(tensor_eye, tensor_pred, "dice_generalized")
    assert assertTensorsEqual(get, expect)


def test_single_scale_loss_jacc():
    array_eye = np.identity((3))
    tensor_eye = np.zeros((3, 3, 3, 3))
    tensor_eye[:, :, 0:3, 0:3] = array_eye

    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_pred[:, 0:2, :, :] = array_eye
    num = np.array([6, 6, 6])
    denom = np.array([9, 9, 9]) + np.array([6, 6, 6]) - num

    expect = 1 - (num / denom)
    get = label.single_scale_loss(tensor_eye, tensor_pred, "jaccard")
    assert assertTensorsEqual(get, expect)


def test_single_scale_loss_mean_sq():
    tensor_mask = np.zeros((3, 3, 3, 3))
    tensor_mask[0, 0, 0, 0] = 1

    tensor_pred = np.ones((3, 3, 3, 3))
    expect = np.array([26 / 27, 1.0, 1.0])

    get = label.single_scale_loss(tensor_mask, tensor_pred, "mean-squared")
    assert assertTensorsEqual(get, expect)


def test_single_scale_loss_other():
    tensor_eye = np.zeros((3, 3, 3, 3))

    tensor_pred = np.zeros((3, 3, 3, 3))

    with pytest.raises(ValueError):
        label.single_scale_loss(tensor_eye, tensor_pred, "random")


def test_multi_scale_loss_pred_len():
    tensor_true = np.zeros((3, 3, 3, 3))
    tensor_pred = np.zeros((3, 3, 3))
    with pytest.raises(AssertionError):
        label.multi_scale_loss(
            tensor_true, tensor_pred, loss_type="jaccard", loss_scales=[0, 1, 2]
        )


def test_multi_scale_loss_true_len():
    tensor_true = np.zeros((3, 3, 3))
    tensor_pred = np.zeros((3, 3, 3, 3))
    with pytest.raises(AssertionError):
        label.multi_scale_loss(
            tensor_true, tensor_pred, loss_type="jaccard", loss_scales=[0, 1, 2]
        )


def test_multi_scale_loss_kernel():
    loss_values = np.asarray([1, 2, 3])
    array_eye = np.identity((3))
    tensor_pred = np.zeros((3, 3, 3, 3))
    tensor_eye = np.zeros((3, 3, 3, 3))

    tensor_eye[:, :, 0:3, 0:3] = array_eye
    tensor_pred[:, :, 0, 0] = array_eye
    tensor_eye = tf.convert_to_tensor(tensor_eye, dtype=tf.double)
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.double)
    list_losses = np.array(
        [
            label.single_scale_loss(
                y_true=label.separable_filter3d(tensor_eye, label.gauss_kernel1d(s)),
                y_pred=label.separable_filter3d(tensor_pred, label.gauss_kernel1d(s)),
                loss_type="jaccard",
            )
            for s in loss_values
        ]
    )
    expect = np.mean(list_losses, axis=0)
    get = label.multi_scale_loss(tensor_eye, tensor_pred, "jaccard", loss_values)
    assert assertTensorsEqual(get, expect)


def test_similarity_fn_unknown_loss():
    config = {"name": "random"}
    with pytest.raises(ValueError):
        label.get_dissimilarity_fn(config)


def test_similarity_fn_multi_scale():
    config = {"name": "multi_scale", "multi_scale": "jaccard"}
    assert isinstance(label.get_dissimilarity_fn(config), FunctionType)


def test_similarity_fn_single_scale():
    config = {"name": "multi_scale", "single_scale": "jaccard"}
    assert isinstance(label.get_dissimilarity_fn(config), FunctionType)
