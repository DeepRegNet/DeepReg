"""
Tests for deepreg/model/loss/image.py in
pytest style
"""
import numpy as np
import pytest
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
    tensor_true = tf.convert_to_tensor(tensor_true, dtype=tf.float32)
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    name_ncc = "lncc"
    get_ncc = image.dissimilarity_fn(tensor_true, tensor_pred, name_ncc)
    expect_ncc = [-0.68002254, -0.9608879]

    tensor_true1 = np.zeros((2, 1, 2, 3))
    tensor_pred1 = 0.6 * np.ones((2, 1, 2, 3))
    tensor_true1 = tf.convert_to_tensor(tensor_true1, dtype=tf.float32)
    tensor_pred1 = tf.convert_to_tensor(tensor_pred1, dtype=tf.float32)

    name_ssd = "ssd"
    get_ssd = image.dissimilarity_fn(tensor_true1, tensor_pred1, name_ssd)
    expect_ssd = [0.36, 0.36]

    get_zero_similarity_ssd = image.dissimilarity_fn(
        tensor_true1, tensor_true1, name_ssd
    )
    get_zero_similarity_ncc = image.dissimilarity_fn(
        tensor_pred1, tensor_pred1, name_ncc
    )

    assert assertTensorsEqual(get_ncc, expect_ncc)
    assert assertTensorsEqual(get_ssd, expect_ssd)
    assert assertTensorsEqual(get_zero_similarity_ssd, [0, 0])
    assert assertTensorsEqual(get_zero_similarity_ncc, [-1, -1])
    with pytest.raises(ValueError):
        image.dissimilarity_fn(
            tensor_true1, tensor_pred1, "some random string that isn't ssd or lncc"
        )


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
