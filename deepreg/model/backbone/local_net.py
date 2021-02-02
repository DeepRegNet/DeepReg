# coding=utf-8

from typing import List

import tensorflow as tf
import tensorflow.keras.layers as tfkl

import deepreg.model.layer_util
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
        use_additive_upsampling: bool = True,
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
        self._use_additive_upsampling = use_additive_upsampling
        self._extract_max_level = max(self._extract_levels)  # E
        self._extract_min_level = min(self._extract_levels)  # D

        # init layer variables
        num_channels = [
            num_channel_initial * (2 ** level)
            for level in range(self._extract_max_level + 1)
        ]  # level 0 to E
        kernel_sizes = [
            7 if level == 0 else 3 for level in range(self._extract_max_level + 1)
        ]
        self._downsample_convs = []
        self._downsample_pools = []
        tensor_shape = image_size
        self._tensor_shapes = [tensor_shape]
        for i in range(self._extract_max_level):
            downsample_conv = tf.keras.Sequential(
                [
                    layer.Conv3dBlock(
                        filters=num_channels[i],
                        kernel_size=kernel_sizes[i],
                        padding="same",
                    ),
                    layer.ResidualConv3dBlock(
                        filters=num_channels[i],
                        kernel_size=kernel_sizes[i],
                        padding="same",
                    ),
                ]
            )
            downsample_pool = tfkl.MaxPool3D(pool_size=2, strides=2, padding="same")
            tensor_shape = tuple((x + 1) // 2 for x in tensor_shape)
            self._downsample_convs.append(downsample_conv)
            self._downsample_pools.append(downsample_pool)
            self._tensor_shapes.append(tensor_shape)

        self._conv3d_block = layer.Conv3dBlock(
            filters=num_channels[-1], kernel_size=3, padding="same"
        )  # level E

        self._upsample_deconvs = []
        self._resizes = []
        self._upsample_convs = []
        for level in range(
            self._extract_max_level - 1, self._extract_min_level - 1, -1
        ):  # level D to E-1
            padding = deepreg.model.layer_util.deconv_output_padding(
                input_shape=self._tensor_shapes[level + 1],
                output_shape=self._tensor_shapes[level],
                kernel_size=kernel_sizes[level],
                stride=2,
                padding="same",
            )
            upsample_deconv = layer.Deconv3dBlock(
                filters=num_channels[level],
                output_padding=padding,
                kernel_size=3,
                strides=2,
                padding="same",
            )
            upsample_conv = tf.keras.Sequential(
                [
                    layer.Conv3dBlock(
                        filters=num_channels[level], kernel_size=3, padding="same"
                    ),
                    layer.ResidualConv3dBlock(
                        filters=num_channels[level], kernel_size=3, padding="same"
                    ),
                ]
            )
            self._upsample_deconvs.append(upsample_deconv)
            self._upsample_convs.append(upsample_conv)
            if self._use_additive_upsampling:
                resize = layer.Resize3d(shape=self._tensor_shapes[level])
                self._resizes.append(resize)
        self._extract_layers = [
            tf.keras.Sequential(
                [
                    tfkl.Conv3D(
                        filters=out_channels,
                        kernel_size=3,
                        strides=1,
                        padding="same",
                        kernel_initializer=out_kernel_initializer,
                        activation=out_activation,
                    ),
                    layer.Resize3d(shape=image_size),
                ]
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
        # outputs used for decoding, encoded[i] corresponds -> level i
        # stored only 0 to E-1
        encoded = []
        h_in = inputs
        for level in range(self._extract_max_level):  # level 0 to E - 1
            skip = self._downsample_convs[level](inputs=h_in, training=training)
            h_in = self._downsample_pools[level](inputs=skip, training=training)
            encoded.append(skip)
        h_bottom = self._conv3d_block(
            inputs=h_in, training=training
        )  # level E of encoding/decoding

        # up sample from level E to D
        decoded = [h_bottom]  # level E
        for idx, level in enumerate(
            range(self._extract_max_level - 1, self._extract_min_level - 1, -1)
        ):  # level E-1 to D
            h = self._upsample_deconvs[idx](inputs=h_bottom, training=training)
            if self._use_additive_upsampling:
                up_sampled = self._resizes[idx](inputs=h_bottom)
                up_sampled = tf.split(up_sampled, num_or_size_splits=2, axis=4)
                up_sampled = tf.add_n(up_sampled)
                h = h + up_sampled
            h = h + encoded[level]
            h_bottom = self._upsample_convs[idx](inputs=h, training=training)
            decoded.append(h_bottom)

        # output
        output = tf.add_n(
            [
                self._extract_layers[idx](
                    inputs=decoded[self._extract_max_level - level]
                )
                for idx, level in enumerate(self._extract_levels)
            ]
        ) / len(self._extract_levels)

        if self.resize:
            output = self.resize(output)
            output = self.interpolate(output)

        return output
