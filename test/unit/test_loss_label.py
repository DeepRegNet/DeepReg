# coding=utf-8

"""
Tests for deepreg/model/loss/label.py in
pytest style
"""
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
    pass


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
    # kernel = np.empty((0))
    # array_eye = np.identity((3))
    # get = label.separable_filter3d(array_eye, kernel)
    # expect = array_eye
    # assert assertTensorsEqual(get, expect)
    pass


def test_compute_centroid():
    pass
    # array_ones = np.ones((2, 2))
    # tensor_mask = np.zeros((3, 2, 2, 2))
    # tensor_mask[0, :, :, :] = array_ones
    # tensor_mask = tf.convert_to_tensor(tensor_mask, dtype=tf.float64)

    # tensor_grid = np.zeros((2, 2, 2, 3))
    # tensor_grid[:, :, :, 0] = array_ones
    # tensor_grid = tf.convert_to_tensor(tensor_grid, dtype=tf.float64)

    # expect = np.array([1, 1, 1])
    # get = label.compute_centroid(tensor_mask, tensor_grid)
    # assert assertTensorsEqual(get, expect)


def test_compute_centroid_d():
    pass


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
    pass


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


def test_single_scale_loss_other():
    tensor_eye = np.zeros((3, 3, 3, 3))

    tensor_pred = np.zeros((3, 3, 3, 3))

    with pytest.raises(ValueError):
        label.single_scale_loss(tensor_eye, tensor_pred, "random")


def test_multi_scale_loss():
    pass


def test_similarity_fn():
    pass
