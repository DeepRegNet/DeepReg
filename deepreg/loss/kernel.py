import math

import tensorflow as tf


def rectangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D rectangular kernel for LocalNormalizedCrossCorrelation.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """

    kernel = tf.ones(shape=(kernel_size,), dtype=tf.float32)
    return kernel


def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D triangular kernel for LocalNormalizedCrossCorrelation.

    Assume kernel_size is odd, it will be a smoothed from
    a kernel which center part is zero.
    Then length of the ones will be around half kernel_size.
    The weight scale of the kernel does not matter as LNCC will normalize it.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """
    assert kernel_size >= 3
    assert kernel_size % 2 != 0

    padding = kernel_size // 2
    kernel = tf.constant(
        [0] * math.ceil(padding / 2)
        + [1] * (kernel_size - padding)
        + [0] * math.floor(padding / 2),
        dtype=tf.float32,
    )

    # (padding*2, )
    filters = tf.ones(shape=(kernel_size - padding, 1, 1), dtype=tf.float32)

    # (kernel_size, 1, 1)
    kernel = tf.nn.conv1d(
        kernel[None, :, None], filters=filters, stride=[1, 1, 1], padding="SAME"
    )

    return kernel[0, :, 0]


def gaussian_kernel1d_size(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D Gaussian kernel for LocalNormalizedCrossCorrelation.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: filters, of shape (kernel_size, )
    """
    mean = (kernel_size - 1) / 2.0
    sigma = kernel_size / 3

    grid = tf.range(0, kernel_size, dtype=tf.float32)
    filters = tf.exp(-tf.square(grid - mean) / (2 * sigma ** 2))

    return filters


def gaussian_kernel1d_sigma(sigma: int) -> tf.Tensor:
    """
    Calculate a gaussian kernel.

    :param sigma: number defining standard deviation for
                  gaussian kernel.
    :return: shape = (dim, )
    """
    assert sigma > 0
    tail = int(sigma * 3)
    kernel = tf.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
    kernel = kernel / tf.reduce_sum(kernel)
    return kernel


def cauchy_kernel1d(sigma: int) -> tf.Tensor:
    """
    Approximating cauchy kernel in 1d.

    :param sigma: int, defining standard deviation of kernel.
    :return: shape = (dim, )
    """
    assert sigma > 0
    tail = int(sigma * 5)
    k = tf.math.reciprocal([((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)])
    k = k / tf.reduce_sum(k)
    return k
