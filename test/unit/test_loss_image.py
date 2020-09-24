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


def test_dissimilarity_fn():
    """
    Testing computed dissimilarity function by comparing to precomputed, the dissimilarity function can be either normalized cross correlation or sum square error function.
    """

    # lncc diff images
    tensor_true = np.array(range(12)).reshape((2, 1, 2, 3))
    tensor_pred = 0.6 * np.ones((2, 1, 2, 3))
    tensor_true = tf.convert_to_tensor(tensor_true, dtype=tf.float32)
    tensor_pred = tf.convert_to_tensor(tensor_pred, dtype=tf.float32)

    name_ncc = "lncc"
    get_ncc = image.dissimilarity_fn(tensor_true, tensor_pred, name_ncc)
    expect_ncc = [-0.68002254, -0.9608879]

    assert is_equal_tf(get_ncc, expect_ncc)

    # ssd diff images
    tensor_true1 = np.zeros((2, 1, 2, 3))
    tensor_pred1 = 0.6 * np.ones((2, 1, 2, 3))
    tensor_true1 = tf.convert_to_tensor(tensor_true1, dtype=tf.float32)
    tensor_pred1 = tf.convert_to_tensor(tensor_pred1, dtype=tf.float32)

    name_ssd = "ssd"
    get_ssd = image.dissimilarity_fn(tensor_true1, tensor_pred1, name_ssd)
    expect_ssd = [0.36, 0.36]

    assert is_equal_tf(get_ssd, expect_ssd)

    # TODO gmi diff images

    # lncc same image
    get_zero_similarity_ncc = image.dissimilarity_fn(
        tensor_pred1, tensor_pred1, name_ncc
    )
    assert is_equal_tf(get_zero_similarity_ncc, [-1, -1])

    # ssd same image
    get_zero_similarity_ssd = image.dissimilarity_fn(
        tensor_true1, tensor_true1, name_ssd
    )
    assert is_equal_tf(get_zero_similarity_ssd, [0, 0])

    # gmi same image
    t = tf.ones([4, 3, 3, 3])
    get_zero_similarity_gmi = image.dissimilarity_fn(t, t, "gmi")
    assert is_equal_tf(get_zero_similarity_gmi, [0, 0, 0, 0])

    # unknown func name
    with pytest.raises(AssertionError):
        image.dissimilarity_fn(
            tensor_true1, tensor_pred1, "some random string that isn't ssd or lncc"
        )


def test_local_normalized_cross_correlation():
    """
    Testing computed local normalized cross correlation function between images using image.local_normalized_cross_correlation by comparing to precomputed.
    """
    tensor_true = np.array(range(24)).reshape((2, 1, 2, 3, 2))
    tensor_pred = 0.6 * np.ones((2, 1, 2, 3, 2))
    expect = [0.7281439, 0.9847701]
    get = image.local_normalized_cross_correlation(
        tensor_true, tensor_pred, kernel_size=9
    )
    assert is_equal_tf(get, expect)


def test_ssd():
    """
    Testing computed sum squared error function between images using image.ssd by comparing to precomputed.
    """
    tensor_true = 0.3 * np.array(range(108)).reshape((2, 3, 3, 3, 2))
    tensor_pred = 0.1 * np.ones((2, 3, 3, 3, 2))
    tensor_pred[:, :, :, :, :] = 1
    get = image.ssd(tensor_true, tensor_pred)
    expect = [70.165, 557.785]
    assert is_equal_tf(get, expect)


def test_gmi():
    """
    Testing computed global mutual information between images using image.global_mutual_information by comparing to precomputed.
    """
    # fixed non trival value
    t1 = np.array(range(108)).reshape((4, 3, 3, 3, 1)) / 108.0
    t1 = tf.convert_to_tensor(t1, dtype=tf.float32)
    t2 = t1 + 0.05
    get = image.global_mutual_information(t1, t2)
    expect = tf.constant(
        [0.84280217, 0.84347117, 0.8441777, 0.8128618], dtype=tf.float32
    )
    assert is_equal_tf(get, expect)

    # zero values
    t1 = tf.zeros((4, 3, 3, 3, 1), dtype=tf.float32)
    t2 = t1
    get = image.global_mutual_information(t1, t2)
    expect = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    assert is_equal_tf(get, expect)

    # zero value and negative value
    t1 = tf.zeros((4, 3, 3, 3, 1), dtype=tf.float32)
    t2 = t1 - 1.0  # will be clipped to zero
    get = image.global_mutual_information(t1, t2)
    expect = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    assert is_equal_tf(get, expect)

    # one values
    t1 = tf.ones((4, 3, 3, 3, 1), dtype=tf.float32)
    t2 = t1
    get = image.global_mutual_information(t1, t2)
    expect = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    assert is_equal_tf(get, expect)
