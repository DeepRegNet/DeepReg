# coding=utf-8

from typing import List, Optional, Tuple, Union

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils import conv_utils

from deepreg.model import layer, layer_util
from deepreg.model.backbone.interface import Backbone
from deepreg.model.layer import Extraction
from deepreg.registry import REGISTRY


@REGISTRY.register_backbone(name="unet")
class UNet(Backbone):
    """
    Class that implements an adapted 3D UNet.

    Reference:

    - O. Ronneberger, P. Fischer, and T. Brox,
      “U-net: Convolutional networks for biomedical image segmentation,”,
      Lecture Notes in Computer Science, 2015, vol. 9351, pp. 234–241.
      https://arxiv.org/abs/1505.04597
    """

    def __init__(
        self,
        image_size: tuple,
        num_channel_initial: int,
        depth: int,
        out_kernel_initializer: str,
        out_activation: str,
        out_channels: int,
        extract_levels: Tuple = (0,),
        pooling: bool = True,
        concat_skip: bool = False,
        encode_kernel_sizes: Union[int, List[int]] = 3,
        decode_kernel_sizes: Union[int, List[int]] = 3,
        encode_num_channels: Optional[Tuple] = None,
        decode_num_channels: Optional[Tuple] = None,
        strides: int = 2,
        padding: str = "same",
        name: str = "Unet",
        **kwargs,
    ):
        """
        Initialise UNet.

        :param image_size: (dim1, dim2, dim3), dims of input image.
        :param num_channel_initial: number of initial channels
        :param depth: input is at level 0, bottom is at level depth.
        :param out_kernel_initializer: kernel initializer for the last layer
        :param out_activation: activation at the last layer
        :param out_channels: number of channels for the output
        :param extract_levels: list, which levels from net to extract.
        :param pooling: for down-sampling, use non-parameterized
                        pooling if true, otherwise use conv3d
        :param concat_skip: when up-sampling, concatenate skipped
                            tensor if true, otherwise use addition
        :param encode_kernel_sizes: kernel size for down-sampling
        :param decode_kernel_sizes: kernel size for up-sampling
        :param encode_num_channels: filters/channels for down-sampling,
            by default it is doubled at each layer during down-sampling
        :param decode_num_channels: filters/channels for up-sampling,
            by default it is the same as encode_num_channels
        :param strides: strides for down-sampling
        :param padding: padding mode for all conv layers
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
        assert max(extract_levels) <= depth
        self._extract_levels = extract_levels
        self._depth = depth

        # save extra parameters
        self._concat_skip = concat_skip
        self._pooling = pooling
        self._encode_kernel_sizes = encode_kernel_sizes
        self._decode_kernel_sizes = decode_kernel_sizes
        self._encode_num_channels = encode_num_channels
        self._decode_num_channels = decode_num_channels
        self._strides = strides
        self._padding = padding

        # init layers
        # all lists start with d = 0
        self._encode_convs: List[tfkl.Layer] = []
        self._encode_pools: List[tfkl.Layer] = []
        self._bottom_block = None
        self._decode_deconvs: List[tfkl.Layer] = []
        self._decode_convs: List[tfkl.Layer] = []
        self._output_block = None

        # build layers
        self.build_layers(
            image_size=image_size,
            num_channel_initial=num_channel_initial,
            depth=depth,
            extract_levels=extract_levels,
            encode_kernel_sizes=encode_kernel_sizes,
            decode_kernel_sizes=decode_kernel_sizes,
            encode_num_channels=encode_num_channels,
            decode_num_channels=decode_num_channels,
            strides=strides,
            padding=padding,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
        )

    def build_encode_conv_block(
        self, filters: int, kernel_size: int, padding: str
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a conv block for down-sampling

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
        self, filters: int, kernel_size: int, padding: str, strides: int
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for down-sampling.

        This block changes the tensor shape (width, height, depth),
        but it does not changes the number of channels.

        :param filters: number of channels for output, arg for conv3d
        :param kernel_size: arg for pool3d or conv3d
        :param padding: arg for pool3d or conv3d
        :param strides: arg for pool3d or conv3d
        :return: a block consists of one or multiple layers
        """
        if self._pooling:
            return tfkl.MaxPool3D(
                pool_size=kernel_size, strides=strides, padding=padding
            )
        else:
            return layer.Conv3dBlock(
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
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
        if self._concat_skip:
            return tfkl.Concatenate()
        else:
            return tfkl.Add()

    def build_decode_conv_block(
        self, filters: int, kernel_size: int, padding: str
    ) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a conv block for up-sampling

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

    def build_layers(
        self,
        image_size: tuple,
        num_channel_initial: int,
        depth: int,
        extract_levels: Tuple[int, ...],
        encode_kernel_sizes: Union[int, List[int]],
        decode_kernel_sizes: Union[int, List[int]],
        encode_num_channels: Optional[Tuple],
        decode_num_channels: Optional[Tuple],
        strides: int,
        padding: str,
        out_kernel_initializer: str,
        out_activation: str,
        out_channels: int,
    ):
        """
        Build layers that will be used in call.

        :param image_size: (dim1, dim2, dim3).
        :param num_channel_initial: number of initial channels.
        :param depth: network starts with d = 0, and the bottom has d = depth.
        :param extract_levels: from which depths the output will be built.
        :param encode_kernel_sizes: kernel size for down-sampling
        :param decode_kernel_sizes: kernel size for up-sampling
        :param encode_num_channels: filters/channels for down-sampling,
            by default it is doubled at each layer during down-sampling
        :param decode_num_channels: filters/channels for up-sampling,
            by default it is the same as encode_num_channels
        :param strides: strides for down-sampling
        :param padding: padding mode for all conv layers
        :param out_kernel_initializer: initializer to use for kernels.
        :param out_activation: activation to use at end layer.
        :param out_channels: number of channels for the extractions
        """
        if encode_num_channels is None:
            assert num_channel_initial >= 1
            encode_num_channels = tuple(
                num_channel_initial * (2 ** d) for d in range(depth + 1)
            )
        assert len(encode_num_channels) == depth + 1
        if decode_num_channels is None:
            decode_num_channels = encode_num_channels
        assert len(decode_num_channels) == depth + 1
        if not self._concat_skip:
            # in case of adding skip tensors, the channels should match
            if decode_num_channels != encode_num_channels:
                raise ValueError(
                    "For UNet, if the skipped tensor is added "
                    "instead of being concatenated, "
                    "the encode_num_channels and decode_num_channels "
                    "should be the same. "
                    f"But got encode_num_channels = {encode_num_channels},"
                    f"decode_num_channels = {decode_num_channels}."
                )
        tensor_shapes = self.build_encode_layers(
            image_size=image_size,
            num_channels=encode_num_channels,
            depth=depth,
            encode_kernel_sizes=encode_kernel_sizes,
            strides=strides,
            padding=padding,
        )
        self.build_decode_layers(
            tensor_shapes=tensor_shapes,
            image_size=image_size,
            num_channels=decode_num_channels,
            depth=depth,
            extract_levels=extract_levels,
            decode_kernel_sizes=decode_kernel_sizes,
            strides=strides,
            padding=padding,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            out_channels=out_channels,
        )

    def build_encode_layers(
        self,
        image_size: Tuple,
        num_channels: Tuple,
        depth: int,
        encode_kernel_sizes: Union[int, List[int]],
        strides: int,
        padding: str,
    ) -> List[Tuple]:
        """
        Build layers for encoding.

        :param image_size: (dim1, dim2, dim3).
        :param num_channels: number of channels for each layer,
            starting from the top layer.
        :param depth: network starts with d = 0, and the bottom has d = depth.
        :param encode_kernel_sizes: kernel size for down-sampling
        :param strides: strides for down-sampling
        :param padding: padding mode for all conv layers
        :return: list of tensor shapes starting from d = 0
        """
        if isinstance(encode_kernel_sizes, int):
            encode_kernel_sizes = [encode_kernel_sizes] * (depth + 1)
        assert len(encode_kernel_sizes) == depth + 1

        # encoding / down-sampling
        self._encode_convs = []
        self._encode_pools = []
        tensor_shape = image_size
        tensor_shapes = [tensor_shape]
        for d in range(depth):
            encode_conv = self.build_encode_conv_block(
                filters=num_channels[d],
                kernel_size=encode_kernel_sizes[d],
                padding=padding,
            )
            encode_pool = self.build_down_sampling_block(
                filters=num_channels[d],
                kernel_size=strides,
                strides=strides,
                padding=padding,
            )
            tensor_shape = tuple(
                conv_utils.conv_output_length(
                    input_length=x,
                    filter_size=strides,
                    padding=padding,
                    stride=strides,
                    dilation=1,
                )
                for x in tensor_shape
            )
            self._encode_convs.append(encode_conv)
            self._encode_pools.append(encode_pool)
            tensor_shapes.append(tensor_shape)

        # bottom layer
        self._bottom_block = self.build_bottom_block(
            filters=num_channels[depth],
            kernel_size=encode_kernel_sizes[depth],
            padding=padding,
        )
        return tensor_shapes

    def build_decode_layers(
        self,
        tensor_shapes: List[Tuple],
        image_size: Tuple,
        num_channels: Tuple,
        depth: int,
        extract_levels: Tuple[int, ...],
        decode_kernel_sizes: Union[int, List[int]],
        strides: int,
        padding: str,
        out_kernel_initializer: str,
        out_activation: str,
        out_channels: int,
    ):
        """
        Build layers for decoding.

        :param tensor_shapes: shapes calculated in encoder
        :param image_size: (dim1, dim2, dim3).
        :param num_channels: number of channels for each layer,
            starting from the top layer.
        :param depth: network starts with d = 0, and the bottom has d = depth.
        :param extract_levels: from which depths the output will be built.
        :param decode_kernel_sizes: kernel size for up-sampling
        :param strides: strides for down-sampling
        :param padding: padding mode for all conv layers
        :param out_kernel_initializer: initializer to use for kernels.
        :param out_activation: activation to use at end layer.
        :param out_channels: number of channels for the extractions
        """
        # init params
        min_extract_level = min(extract_levels)
        if isinstance(decode_kernel_sizes, int):
            decode_kernel_sizes = [decode_kernel_sizes] * depth
        assert len(decode_kernel_sizes) == depth

        # decoding / up-sampling
        self._decode_deconvs = []
        self._decode_convs = []
        for d in range(depth - 1, min_extract_level - 1, -1):
            kernel_size = decode_kernel_sizes[d]
            output_padding = layer_util.deconv_output_padding(
                input_shape=tensor_shapes[d + 1],
                output_shape=tensor_shapes[d],
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
            )
            decode_deconv = self.build_up_sampling_block(
                filters=num_channels[d],
                output_padding=output_padding,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                output_shape=tensor_shapes[d],
            )
            decode_conv = self.build_decode_conv_block(
                filters=num_channels[d], kernel_size=kernel_size, padding=padding
            )
            self._decode_deconvs = [decode_deconv] + self._decode_deconvs
            self._decode_convs = [decode_conv] + self._decode_convs
        if min_extract_level > 0:
            # add Nones to make lists have length depth - 1
            self._decode_deconvs = [None] * min_extract_level + self._decode_deconvs
            self._decode_convs = [None] * min_extract_level + self._decode_convs

        # extraction
        self._output_block = self.build_output_block(
            image_size=image_size,
            extract_levels=extract_levels,
            out_channels=out_channels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """
        Build compute graph based on built layers.

        :param inputs: image batch, shape = (batch, f_dim1, f_dim2, f_dim3, ch)
        :param training: None or bool.
        :param mask: None or tf.Tensor.
        :return: shape = (batch, f_dim1, f_dim2, f_dim3, out_channels)
        """

        # encoding / down-sampling
        skips = []
        encoded = inputs
        for d in range(self._depth):
            skip = self._encode_convs[d](inputs=encoded, training=training)
            encoded = self._encode_pools[d](inputs=skip, training=training)
            skips.append(skip)

        # bottom
        decoded = self._bottom_block(inputs=encoded, training=training)  # type: ignore

        # decoding / up-sampling
        outs = [decoded]
        for d in range(self._depth - 1, min(self._extract_levels) - 1, -1):
            decoded = self._decode_deconvs[d](inputs=decoded, training=training)
            decoded = self.build_skip_block()([decoded, skips[d]])
            decoded = self._decode_convs[d](inputs=decoded, training=training)
            outs = [decoded] + outs

        # output
        output = self._output_block(outs)  # type: ignore

        return output

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(
            depth=self._depth,
            extract_levels=self._extract_levels,
            pooling=self._pooling,
            concat_skip=self._concat_skip,
            encode_kernel_sizes=self._encode_kernel_sizes,
            decode_kernel_sizes=self._decode_kernel_sizes,
            encode_num_channels=self._encode_num_channels,
            decode_num_channels=self._decode_num_channels,
            strides=self._strides,
            padding=self._padding,
        )
        return config
