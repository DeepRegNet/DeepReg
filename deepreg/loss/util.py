"""Provide helper functions or classes for defining loss or metrics."""

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
