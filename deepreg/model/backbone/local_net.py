# coding=utf-8

from typing import List

import tensorflow as tf

from deepreg.model import layer
from deepreg.model.backbone.interface import Backbone
from deepreg.registry import REGISTRY


@REGISTRY.register_backbone(name="local")
class LocalNet(Backbone):
    """
    Build LocalNet for image registration.

    Reference:

    - Hu, Yipeng, et al.
      "Weakly-supervised convolutional neural networks
      for multimodal image registration."
      Medical image analysis 49 (2018): 1-13.
      https://doi.org/10.1016/j.media.2018.07.002

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
        extract_levels: List[int],
        out_kernel_initializer: str,
        out_activation: str,
        control_points: (tuple, None) = None,
        name: str = "LocalNet",
        **kwargs,
    ):
        """
        Image is encoded gradually, i from level 0 to E,
        then it is decoded gradually, j from level E to D.
        Some of the decoded levels are used for generating extractions.

        So, extract_levels are between [0, E] with E = max(extract_levels),
        and D = min(extract_levels).

        :param image_size: such as (dim1, dim2, dim3)
        :param out_channels: number of channels for the extractions
        :param num_channel_initial: number of initial channels.
        :param extract_levels: number of extraction levels.
        :param out_kernel_initializer: initializer to use for kernels.
        :param out_activation: activation to use at end layer.
        :param control_points: specify the distance between control points (in voxels).
        :param name: name of the backbone.
        :param kwargs: additional arguments.
        """
        super().__init__(
            image_size=image_size,
            out_channels=out_channels,
            num_channel_initial=num_channel_initial,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            name=name,
            **kwargs,
        )

        # save parameters
        self._extract_levels = extract_levels
        self._extract_max_level = max(self._extract_levels)  # E
        self._extract_min_level = min(self._extract_levels)  # D

        # init layer variables
        num_channels = [
            num_channel_initial * (2 ** level)
            for level in range(self._extract_max_level + 1)
        ]  # level 0 to E
        self._downsample_blocks = [
            layer.DownSampleResnetBlock(
                filters=num_channels[i], kernel_size=7 if i == 0 else 3
            )
            for i in range(self._extract_max_level)
        ]  # level 0 to E-1
        self._conv3d_block = layer.Conv3dBlock(filters=num_channels[-1])  # level E

        self._upsample_blocks = [
            layer.LocalNetUpSampleResnetBlock(num_channels[level])
            for level in range(
                self._extract_max_level - 1, self._extract_min_level - 1, -1
            )
        ]  # level D to E-1

        self._extract_layers = [
            # if kernels are not initialized by zeros, with init NN, extract may be too large
            layer.Conv3dWithResize(
                output_shape=image_size,
                filters=out_channels,
                kernel_initializer=out_kernel_initializer,
                activation=out_activation,
            )
            for _ in self._extract_levels
        ]

        self.resize = (
            layer.ResizeCPTransform(control_points)
            if control_points is not None
            else False
        )
        self.interpolate = (
            layer.BSplines3DTransform(control_points, image_size)
            if control_points is not None
            else False
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """
        Build LocalNet graph based on built layers.

        :param inputs: image batch, shape = (batch, f_dim1, f_dim2, f_dim3, ch)
        :param training: None or bool.
        :param mask: None or tf.Tensor.
        :return: shape = (batch, f_dim1, f_dim2, f_dim3, out_channels)
        """

        # down sample from level 0 to E
        encoded = []
        # outputs used for decoding, encoded[i] corresponds -> level i
        # stored only 0 to E-1

        h_in = inputs
        for level in range(self._extract_max_level):  # level 0 to E - 1
            h_in, h_channel = self._downsample_blocks[level](
                inputs=h_in, training=training
            )
            encoded.append(h_channel)
        h_bottom = self._conv3d_block(
            inputs=h_in, training=training
        )  # level E of encoding/decoding

        # up sample from level E to D
        decoded = [h_bottom]  # level E
        for idx, level in enumerate(
            range(self._extract_max_level - 1, self._extract_min_level - 1, -1)
        ):  # level E-1 to D
            h_bottom = self._upsample_blocks[idx](
                inputs=[h_bottom, encoded[level]], training=training
            )
            decoded.append(h_bottom)

        # output
        output = tf.reduce_mean(
            tf.stack(
                [
                    self._extract_layers[idx](
                        inputs=decoded[self._extract_max_level - level]
                    )
                    for idx, level in enumerate(self._extract_levels)
                ],
                axis=5,
            ),
            axis=5,
        )

        if self.resize:
            output = self.resize(output)
            output = self.interpolate(output)

        return output
