"""Provide helper functions or classes for defining loss or metrics."""

from typing import List, Optional, Union

import tensorflow as tf

from deepreg.loss.kernel import cauchy_kernel1d
from deepreg.loss.kernel import gaussian_kernel1d_sigma as gaussian_kernel1d


class MultiScaleMixin(tf.keras.losses.Loss):
    """
    Mixin class for multi-scale loss.

    It applies the loss at different scales (gaussian or cauchy smoothing).
    It is assumed that loss values are between 0 and 1.
    """

    kernel_fn_dict = dict(gaussian=gaussian_kernel1d, cauchy=cauchy_kernel1d)

    def __init__(
        self,
        scales: Optional[Union[List, float, int]] = None,
        kernel: str = "gaussian",
        **kwargs,
    ):
        """
        Init.

        :param scales: list of scalars or None, if None, do not apply any scaling.
        :param kernel: gaussian or cauchy.
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        if kernel not in self.kernel_fn_dict:
            raise ValueError(
                f"Kernel {kernel} is not supported."
                f"Supported kernels are {list(self.kernel_fn_dict.keys())}"
            )
        if scales is not None and not isinstance(scales, list):
            scales = [scales]
        self.scales = scales
        self.kernel = kernel

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Use super().call to calculate loss at different scales.

        :param y_true: ground-truth tensor, shape = (batch, dim1, dim2, dim3).
        :param y_pred: predicted tensor, shape = (batch, dim1, dim2, dim3).
        :return: multi-scale loss, shape = (batch, ).
        """
        if self.scales is None:
            return super().call(y_true=y_true, y_pred=y_pred)
        kernel_fn = self.kernel_fn_dict[self.kernel]
        losses = []
        for s in self.scales:
            if s == 0:
                # no smoothing
                losses.append(
                    super().call(
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                )
            else:
                losses.append(
                    super().call(
                        y_true=separable_filter(
                            tf.expand_dims(y_true, axis=4), kernel_fn(s)
                        )[..., 0],
                        y_pred=separable_filter(
                            tf.expand_dims(y_pred, axis=4), kernel_fn(s)
                        )[..., 0],
                    )
                )
        loss = tf.add_n(losses)
        loss = loss / len(self.scales)
        return loss

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["scales"] = self.scales
        config["kernel"] = self.kernel
        return config


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
