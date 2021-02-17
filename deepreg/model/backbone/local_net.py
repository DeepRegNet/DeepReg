# coding=utf-8

from typing import List, Tuple, Union

import tensorflow as tf
import tensorflow.keras.layers as tfkl

from deepreg.model import layer
from deepreg.model.backbone.u_net import AbstractUNet
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


class Extraction(tfkl.Layer):
    def __init__(
        self,
        image_size: Tuple[int],
        extract_levels: List[int],
        out_channels: int,
        out_kernel_initializer: str,
        out_activation: str,
        name: str = "Extraction",
    ):
        """
        :param image_size: such as (dim1, dim2, dim3)
        :param extract_levels: number of extraction levels.
        :param out_channels: number of channels for the extractions
        :param out_kernel_initializer: initializer to use for kernels.
        :param out_activation: activation to use at end layer.
        :param name: name of the layer
        """
        super().__init__(name=name)
        self.extract_levels = extract_levels
        self.max_level = max(extract_levels)
        self.layers = [
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
            for _ in extract_levels
        ]

    def call(self, inputs: List[tf.Tensor], **kwargs) -> tf.Tensor:
        """

        :param inputs: a list of tensors
        :param kwargs:
        :return:
        """

        return tf.add_n(
            [
                self.layers[idx](inputs=inputs[self.max_level - level])
                for idx, level in enumerate(self.extract_levels)
            ]
        ) / len(self.extract_levels)


@REGISTRY.register_backbone(name="local")
class LocalNet(AbstractUNet):
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
        num_channel_initial: int,
        extract_levels: List[int],
        out_kernel_initializer: str,
        out_activation: str,
        out_channels: int,
        use_additive_upsampling: bool = True,
        name: str = "LocalNet",
        **kwargs,
    ):
        """
        Init.

        Image is encoded gradually, i from level 0 to D,
        then it is decoded gradually, j from level D to 0.
        Some of the decoded levels are used for generating extractions.

        So, extract_levels are between [0, D].

        :param image_size: such as (dim1, dim2, dim3)
        :param num_channel_initial: number of initial channels.
        :param extract_levels: from which depths the output will be built.
        :param out_kernel_initializer: initializer to use for kernels.
        :param out_activation: activation to use at end layer.
        :param out_channels: number of channels for the extractions
        :param use_additive_upsampling: whether use additive up-sampling.
        :param name: name of the backbone.
        :param kwargs: additional arguments.
        """
        super().__init__(
            image_size=image_size,
            num_channel_initial=num_channel_initial,
            depth=max(extract_levels),
            extract_levels=extract_levels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
            name=name,
            **kwargs,
        )

        # save extra parameters
        self._use_additive_upsampling = use_additive_upsampling

        # build layers
        self.build_layers(
            image_size=image_size,
            num_channel_initial=num_channel_initial,
            depth=self._depth,
            extract_levels=self._extract_levels,
            downsample_kernel_sizes=[7] + [3] * self._depth,
            upsample_kernel_sizes=3,
            strides=2,
            padding="same",
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
        )

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

    def build_output_block(
        self,
        image_size: Tuple[int],
        extract_levels: List[int],
        out_channels: int,
        out_kernel_initializer: str,
        out_activation: str,
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for output.

        The input to this block is a list of tensors.

        :param image_size: such as (dim1, dim2, dim3)
        :param extract_levels: number of extraction levels.
        :param out_channels: number of channels for the extractions
        :param out_kernel_initializer: initializer to use for kernels.
        :param out_activation: activation to use at end layer.
        :return: a block consists of one or multiple layers
        """
        return Extraction(
            image_size=image_size,
            extract_levels=extract_levels,
            out_channels=out_channels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
        )
