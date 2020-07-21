"""
Tests for deepreg/model/loss/image.py in
pytest style
"""
import numpy as np
import tensorflow as tf

import deepreg.model.loss.image as image


def assertTensorsEqual(x, y):
    """
    given two tf tensors return True/False (not tf tensor)
    tolerate small errors
    :param x:
    :param y:
    :return:
    """
    return tf.reduce_max(tf.abs(x - y)).numpy() < 1e-6


def test_dissimilarity_fn():
    """
    Testing computed dissimilarity function by comparing to precomputed, the dissimilarity function can be either
    normalized cross correlation or sum square error function
    """
    tensor_true = np.array(range(12)).reshape((2, 1, 2, 3))
    tensor_pred = 0.6 * np.ones((2, 1, 2, 3))

    tensor_true_expand = tf.expand_dims(tensor_true, axis=4)
    tensor_pred_expand = tf.expand_dims(tensor_pred, axis=4)
    tensor_true_expand_np = np.array(tensor_true_expand)
    tensor_pred_expand_np = np.array(tensor_pred_expand)

    get_ncc = image.local_normalized_cross_correlation(
        tensor_true_expand_np, tensor_pred_expand_np, kernel_size=9
    )
    expect_ncc = [0.68002254, 0.9608879]
    get_ssd = image.ssd(tensor_true_expand_np, tensor_pred_expand_np)
    expect_ssd = [6.52666667, 65.32666667]
    assert assertTensorsEqual(get_ncc, expect_ncc)
    assert assertTensorsEqual(get_ssd, expect_ssd)


def test_local_normalized_cross_correlation():
    """
    Testing computed local normalized cross correlation function by comparing to precomputed
    """
    tensor_true = np.array(range(24)).reshape((2, 1, 2, 3, 2))
    tensor_pred = 0.6 * np.ones((2, 1, 2, 3, 2))
    expect = [0.7281439, 0.9847701]
    get = image.local_normalized_cross_correlation(
        tensor_true, tensor_pred, kernel_size=9
    )
    assert assertTensorsEqual(get, expect)


def test_ssd():
    """
    Testing computed sum squared error function by comparing to precomputed
    """
    tensor_true = 0.3 * np.array(range(108)).reshape((2, 3, 3, 3, 2))
    tensor_pred = 0.1 * np.ones((2, 3, 3, 3, 2))
    tensor_pred[:, :, :, :, :] = 1
    get = image.ssd(tensor_true, tensor_pred)
    expect = [70.165, 557.785]
    assert assertTensorsEqual(get, expect)
