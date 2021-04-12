# coding=utf-8

from typing import List, Optional, Tuple, Union

import tensorflow as tf
import tensorflow.keras.layers as tfkl

from deepreg.model import layer
from deepreg.model.backbone.u_net import UNet
from deepreg.model.layer import Extraction
from deepreg.registry import REGISTRY


class AdditiveUpsampling(tfkl.Layer):
    def __init__(
        self,
        filters: int,
        output_padding: Union[int, Tuple, List],
        kernel_size: Union[int, Tuple, List],
        padding: str,
        strides: Union[int, Tuple, List],
        output_shape: Tuple,
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

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        deconv_config = self.deconv3d.get_config()
        config.update(
            filters=deconv_config["filters"],
            output_padding=deconv_config["output_padding"],
            kernel_size=deconv_config["kernel_size"],
            strides=deconv_config["strides"],
            padding=deconv_config["padding"],
        )
        config.update(output_shape=self.resize._shape)
        return config


@REGISTRY.register_backbone(name="local")
class LocalNet(UNet):
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
        extract_levels: Tuple[int, ...],
        out_kernel_initializer: str,
        out_activation: str,
        out_channels: int,
        depth: Optional[int] = None,
        use_additive_upsampling: bool = True,
        pooling: bool = True,
        concat_skip: bool = False,
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
        :param depth: depth of the encoder.
            If depth is not given, depth = max(extract_levels) will be used.
        :param use_additive_upsampling: whether use additive up-sampling layer
            for decoding.
        :param pooling: for down-sampling, use non-parameterized
                        pooling if true, otherwise use conv3d
        :param concat_skip: when up-sampling, concatenate skipped
                            tensor if true, otherwise use addition
        :param name: name of the backbone.
        :param kwargs: additional arguments.
        """
        self._use_additive_upsampling = use_additive_upsampling
        if depth is None:
            depth = max(extract_levels)
        kwargs["encode_kernel_sizes"] = [7] + [3] * depth
        super().__init__(
            image_size=image_size,
            num_channel_initial=num_channel_initial,
            depth=depth,
            extract_levels=extract_levels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
            pooling=pooling,
            concat_skip=concat_skip,
            name=name,
            **kwargs,
        )

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
        output_padding: Union[Tuple[int, ...], int],
        kernel_size: Union[Tuple[int, ...], int],
        padding: str,
        strides: Union[Tuple[int, ...], int],
        output_shape: Tuple[int, ...],
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

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(use_additive_upsampling=self._use_additive_upsampling)
        return config
