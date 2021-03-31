"""Provide helper functions or classes for defining loss or metrics."""
import math

import tensorflow as tf


class NegativeLossMixin(tf.keras.losses.Loss):
    """Mixin class to revert the sign of the loss value."""

    def __init__(self, **kwargs):
        """
        Init without required arguments.

        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        self.name = self.name + "Loss"

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Revert the sign of loss.

        :param y_true: ground-truth tensor.
        :param y_pred: predicted tensor.
        :return: negated loss.
        """
        return -super().call(y_true=y_true, y_pred=y_pred)


EPS = tf.keras.backend.epsilon()


def rectangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    Return a the 1D filter for separable convolution equivalent to a 3-D rectangular
    kernel for LocalNormalizedCrossCorrelation.

    :param kernel_size: scalar, size of the 1-D kernel
    :return: kernel_weights, of shape (kernel_size, )
    """

    kernel = tf.ones(shape=(kernel_size,), dtype=tf.float32)
    return kernel


def triangular_kernel1d(kernel_size: int) -> tf.Tensor:
    """
    1D triangular kernel.

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
    Return a the 1D filter for separable convolution equivalent to a 3-D Gaussian
    kernel for LocalNormalizedCrossCorrelation.
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


def separable_filter(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """
    Create a 3d separable filter.

    Here `tf.nn.conv3d` accepts the `filters` argument of shape
    (filter_depth, filter_height, filter_width, in_channels, out_channels),
    where the first axis of `filters` is the depth not batch,
    and the input to `tf.nn.conv3d` is of shape
    (batch, in_depth, in_height, in_width, in_channels).

    :param tensor: shape = (batch, dim1, dim2, dim3, 1)
    :param kernel: shape = (dim4,)
    :return: shape = (batch, dim1, dim2, dim3, 1)
    """
    strides = [1, 1, 1, 1, 1]
    kernel = tf.cast(kernel, dtype=tensor.dtype)

    tensor = tf.nn.conv3d(
        tf.nn.conv3d(
            tf.nn.conv3d(
                tensor,
                filters=tf.reshape(kernel, [-1, 1, 1, 1, 1]),
                strides=strides,
                padding="SAME",
            ),
            filters=tf.reshape(kernel, [1, -1, 1, 1, 1]),
            strides=strides,
            padding="SAME",
        ),
        filters=tf.reshape(kernel, [1, 1, -1, 1, 1]),
        strides=strides,
        padding="SAME",
    )
    return tensor
