# coding=utf-8

from typing import List, Union

import tensorflow as tf
import tensorflow.keras.layers as tfkl

from deepreg.model import layer, layer_util
from deepreg.model.backbone.interface import Backbone
from deepreg.registry import REGISTRY


class AdditiveUpsampling(tfkl.Layer):
    def __init__(
        self,
        filters: int,
        output_padding: int,
        kernel_size: int,
        padding: str,
        strides: int,
        output_shape: tuple,
        name: str = "AdditiveUpsampling",
    ):
        """
        Addictive up-sampling layer.

        :param filters: number of channels for output
        :param output_padding: padding for output
        :param kernel_size: arg for deconv3d
        :param padding: arg for deconv3d
        :param strides: arg for deconv3d
        :param output_shape: shape of the output tensor
        :param name: name of the layer.
        """
        super().__init__(name=name)
        self.deconv3d = layer.Deconv3dBlock(
            filters=filters,
            output_padding=output_padding,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )
        self.resize = layer.Resize3d(shape=output_shape)

    def call(self, inputs, **kwargs):
        deconved = self.deconv3d(inputs)
        resized = self.resize(inputs)
        resized = tf.add_n(tf.split(resized, num_or_size_splits=2, axis=4))
        return deconved + resized


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
        :param use_additive_upsampling: whether use additive up-sampling.
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
            downsample_conv = self.build_conv_block(
                filters=num_channels[i], kernel_size=kernel_sizes[i], padding="same"
            )
            downsample_pool = self.build_down_sampling_block(
                kernel_size=2, strides=2, padding="same"
            )
            tensor_shape = tuple((x + 1) // 2 for x in tensor_shape)
            self._downsample_convs.append(downsample_conv)
            self._downsample_pools.append(downsample_pool)
            self._tensor_shapes.append(tensor_shape)

        self._bottom_block = self.build_bottom_block(
            filters=num_channels[-1], kernel_size=3, padding="same"
        )  # level E

        self._upsample_deconvs = []
        self._upsample_convs = []
        for level in range(
            self._extract_max_level - 1, self._extract_min_level - 1, -1
        ):  # level D to E-1
            padding = layer_util.deconv_output_padding(
                input_shape=self._tensor_shapes[level + 1],
                output_shape=self._tensor_shapes[level],
                kernel_size=kernel_sizes[level],
                stride=2,
                padding="same",
            )
            upsample_deconv = self.build_up_sampling_block(
                filters=num_channels[level],
                output_padding=padding,
                kernel_size=3,
                strides=2,
                padding="same",
                output_shape=self._tensor_shapes[level],
            )
            upsample_conv = self.build_conv_block(
                filters=num_channels[level], kernel_size=3, padding="same"
            )
            self._upsample_deconvs.append(upsample_deconv)
            self._upsample_convs.append(upsample_conv)
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

    def build_conv_block(
        self, filters: int, kernel_size: int, padding: str
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a conv block for down-sampling or up-sampling.

        This block do not change the tensor shape (width, height, depth),
        it only changes the number of channels.

        :param filters: number of channels for output
        :param kernel_size: arg for conv3d
        :param padding: arg for conv3d
        :return: a block consists of one or multiple layers
        """
        return tf.keras.Sequential(
            [
                layer.Conv3dBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
                layer.ResidualConv3dBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                ),
            ]
        )

    def build_down_sampling_block(
        self, kernel_size: int, padding: str, strides: int
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for down-sampling.

        This block changes the tensor shape (width, height, depth),
        but it does not changes the number of channels.

        :param kernel_size: arg for pool3d
        :param padding: arg for pool3d
        :param strides: arg for pool3d
        :return: a block consists of one or multiple layers
        """
        return tfkl.MaxPool3D(pool_size=kernel_size, strides=strides, padding=padding)

    def build_bottom_block(
        self, filters: int, kernel_size: int, padding: str
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for bottom layer.

        This block do not change the tensor shape (width, height, depth),
        it only changes the number of channels.

        :param filters: number of channels for output
        :param kernel_size: arg for conv3d
        :param padding: arg for conv3d
        :return: a block consists of one or multiple layers
        """
        return layer.Conv3dBlock(
            filters=filters, kernel_size=kernel_size, padding=padding
        )

    def build_up_sampling_block(
        self,
        filters: int,
        output_padding: int,
        kernel_size: int,
        padding: str,
        strides: int,
        output_shape: tuple,
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for up-sampling.

        This block changes the tensor shape (width, height, depth),
        but it does not changes the number of channels.

        :param filters: number of channels for output
        :param output_padding: padding for output
        :param kernel_size: arg for deconv3d
        :param padding: arg for deconv3d
        :param strides: arg for deconv3d
        :param output_shape: shape of the output tensor
        :return: a block consists of one or multiple layers
        """

        if self._use_additive_upsampling:
            return AdditiveUpsampling(
                filters=filters,
                output_padding=output_padding,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                output_shape=output_shape,
            )

        return layer.Deconv3dBlock(
            filters=filters,
            output_padding=output_padding,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )

    def build_skip_block(self) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for combining skipped tensor and up-sampled one.

        This block do not change the tensor shape (width, height, depth),
        it only changes the number of channels.

        The input to this block is a list of tensors.

        :return: a block consists of one or multiple layers
        """
        return tfkl.Add()

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
        skips = []
        down_sampled = inputs
        for level in range(self._extract_max_level):  # level 0 to E - 1
            skip = self._downsample_convs[level](inputs=down_sampled, training=training)
            down_sampled = self._downsample_pools[level](inputs=skip, training=training)
            skips.append(skip)
        up_sampled = self._bottom_block(
            inputs=down_sampled, training=training
        )  # level E of encoding/decoding

        # up sample from level E to D
        outs = [up_sampled]  # level E
        for idx, level in enumerate(
            range(self._extract_max_level - 1, self._extract_min_level - 1, -1)
        ):  # level E-1 to D
            up_sampled = self._upsample_deconvs[idx](
                inputs=up_sampled, training=training
            )
            up_sampled = self.build_skip_block()([up_sampled, skips[level]])
            up_sampled = self._upsample_convs[idx](inputs=up_sampled, training=training)
            outs.append(up_sampled)

        # output
        output = tf.add_n(
            [
                self._extract_layers[idx](inputs=outs[self._extract_max_level - level])
                for idx, level in enumerate(self._extract_levels)
            ]
        ) / len(self._extract_levels)

        return output
