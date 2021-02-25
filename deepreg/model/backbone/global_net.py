# coding=utf-8

from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from deepreg.model import layer_util
from deepreg.model.backbone.u_net import UNet
from deepreg.registry import REGISTRY


class AffineHead(tfkl.Layer):
    def __init__(
        self,
        image_size: tuple,
        name: str = "AffineHead",
    ):
        """
        Init.

        :param image_size: such as (dim1, dim2, dim3)
        :param name: name of the layer
        """
        super().__init__(name=name)
        self.reference_grid = layer_util.get_reference_grid(image_size)
        self.transform_initial = tf.constant_initializer(
            value=list(np.eye(4, 3).reshape((-1)))
        )
        self._flatten = tfkl.Flatten()
        self._dense = tfkl.Dense(units=12, bias_initializer=self.transform_initial)

    def call(
        self, inputs: Union[tf.Tensor, List], **kwargs
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """

        :param inputs: a tensor or a list of tensor with length 1
        :param kwargs: additional args
        :return: ddf and theta

            - ddf has shape (batch, dim1, dim2, dim3, 3)
            - theta has shape (batch, 4, 3)
        """
        if isinstance(inputs, list):
            inputs = inputs[0]
        theta = self._dense(self._flatten(inputs))
        theta = tf.reshape(theta, shape=(-1, 4, 3))
        # warp the reference grid with affine parameters to output a ddf
        grid_warped = layer_util.warp_grid(self.reference_grid, theta)
        ddf = grid_warped - self.reference_grid
        return ddf, theta

    def get_config(self):
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(image_size=self.reference_grid.shape[:3])
        return config


@REGISTRY.register_backbone(name="global")
class GlobalNet(UNet):
    """
    Build GlobalNet for image registration.

    GlobalNet is a special UNet where the decoder for up-sampling is skipped.
    The network's outputs come from the bottom layer from the encoder directly.

    Reference:

    - Hu, Yipeng, et al.
      "Label-driven weakly-supervised learning
      for multimodal deformable image registration,"
      https://arxiv.org/abs/1711.01666
    """

    def __init__(
        self,
        image_size: tuple,
        num_channel_initial: int,
        extract_levels: Optional[Tuple[int, ...]] = None,
        depth: Optional[int] = None,
        name: str = "GlobalNet",
        **kwargs,
    ):
        """
        Image is encoded gradually, i from level 0 to E.
        Then, a densely-connected layer outputs an affine
        transformation.

        :param image_size: tuple, such as (dim1, dim2, dim3)
        :param num_channel_initial: int, number of initial channels
        :param extract_levels: list, which levels from net to extract, deprecated.
            If depth is not given, depth = max(extract_levels) will be used.
        :param depth: depth of the encoder. If given, extract_levels is not used.
        :param name: name of the backbone.
        :param kwargs: additional arguments.
        """
        if depth is None:
            if extract_levels is None:
                raise ValueError(
                    "GlobalNet requires `depth` or `extract_levels` "
                    "to define the depth of encoder. "
                    "If `depth` is not given, "
                    "the maximum value of `extract_levels` will be used."
                    "However the argument `extract_levels` is deprecated "
                    "and will be removed in future release."
                )
            depth = max(extract_levels)
        super().__init__(
            image_size=image_size,
            num_channel_initial=num_channel_initial,
            depth=depth,
            extract_levels=(depth,),
            name=name,
            **kwargs,
        )

    def build_output_block(
        self,
        image_size: Tuple[int, ...],
        extract_levels: Tuple[int, ...],
        out_channels: int,
        out_kernel_initializer: str,
        out_activation: str,
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for output.

        The input to this block is a list of length 1.
        The output has two tensors.

        :param image_size: such as (dim1, dim2, dim3)
        :param extract_levels: not used
        :param out_channels: not used
        :param out_kernel_initializer: not used
        :param out_activation: not used
        :return: a block consists of one or multiple layers
        """
        return AffineHead(image_size=image_size)
