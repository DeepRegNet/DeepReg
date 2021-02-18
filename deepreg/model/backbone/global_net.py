# coding=utf-8

from typing import Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl

from deepreg.model import layer_util
from deepreg.model.backbone.local_net import LocalNet
from deepreg.registry import REGISTRY


class AffineHead(tfkl.Layer):
    def __init__(
        self,
        image_size: tuple,
        name: str = "AffineHead",
    ):
        super().__init__(name=name)
        self.reference_grid = layer_util.get_reference_grid(image_size)
        self.transform_initial = tf.constant_initializer(
            value=list(np.eye(4, 3).reshape((-1)))
        )
        self._flatten = tfkl.Flatten()
        self._dense = tfkl.Dense(units=12, bias_initializer=self.transform_initial)

    def call(self, inputs, **kwargs):
        theta = self._dense(self._flatten(inputs[0]))
        theta = tf.reshape(theta, shape=(-1, 4, 3))
        # warp the reference grid with affine parameters to output a ddf
        grid_warped = layer_util.warp_grid(self.reference_grid, theta)
        ddf = grid_warped - self.reference_grid
        return ddf, theta


@REGISTRY.register_backbone(name="global")
class GlobalNet(LocalNet):
    """
    Build GlobalNet for image registration.

    Reference:

    - Hu, Yipeng, et al.
      "Label-driven weakly-supervised learning
      for multimodal deformable image registration,"
      https://arxiv.org/abs/1711.01666
    """

    def __init__(
        self,
        image_size: tuple,
        out_channels: int,
        num_channel_initial: int,
        extract_levels: Tuple[int],
        out_kernel_initializer: str,
        out_activation: str,
        depth: Optional[int] = None,
        name: str = "GlobalNet",
        **kwargs,
    ):
        """
        Image is encoded gradually, i from level 0 to E.
        Then, a densely-connected layer outputs an affine
        transformation.

        :param image_size: tuple, such as (dim1, dim2, dim3)
        :param out_channels: int, number of channels for the output
        :param num_channel_initial: int, number of initial channels
        :param extract_levels: list, which levels from net to extract.
        :param out_kernel_initializer: not used
        :param out_activation: not used
        :param depth: depth of the encoder.
        :param name: name of the backbone.
        :param kwargs: additional arguments.
        """
        if depth is None:
            depth = max(extract_levels)
        super().__init__(
            image_size=image_size,
            num_channel_initial=num_channel_initial,
            depth=depth,
            extract_levels=(depth,),
            out_kernel_initializer="",
            out_activation="",
            out_channels=3,
            name=name,
            **kwargs,
        )

    def build_output_block(
        self, image_size: Tuple[int], **kwargs
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for output.

        The input to this block is a list of length 1.
        The output has two tensors.

        :param image_size: such as (dim1, dim2, dim3)
        :param kwargs: unused args
        :return: a block consists of one or multiple layers
        """
        return AffineHead(image_size=image_size)
