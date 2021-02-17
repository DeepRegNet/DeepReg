# coding=utf-8


from abc import abstractmethod
from typing import List, Tuple, Union

import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils import conv_utils

from deepreg.model import layer, layer_util
from deepreg.model.backbone.interface import Backbone
from deepreg.registry import REGISTRY


class AbstractUNet(Backbone):
    """An interface for u-net style backbones."""

    def __init__(
        self,
        image_size: tuple,
        num_channel_initial: int,
        depth: int,
        extract_levels: List[int],
        out_kernel_initializer: str,
        out_activation: str,
        out_channels: int,
        name: str = "AbstractUNet",
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
        :param depth: d = 0 to depth, both side included
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

        # init layers
        # all lists start with d = 0
        self._downsample_convs = None
        self._downsample_pools = None
        self._bottom_block = None
        self._upsample_deconvs = None
        self._upsample_convs = None
        self._output_block = None

    def build_layers(
        self,
        image_size: tuple,
        num_channel_initial: int,
        depth: int,
        extract_levels: List[int],
        downsample_kernel_sizes: Union[int, List[int]],
        upsample_kernel_sizes: Union[int, List[int]],
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
        :param downsample_kernel_sizes: kernel size for down-sampling
        :param upsample_kernel_sizes: kernel size for up-sampling
        :param strides: strides for down-sampling
        :param padding: padding mode for all conv layers
        :param out_kernel_initializer: initializer to use for kernels.
        :param out_activation: activation to use at end layer.
        :param out_channels: number of channels for the extractions
        """
        # init params
        min_extract_level = min(extract_levels)
        num_channels = [num_channel_initial * (2 ** d) for d in range(depth + 1)]
        if isinstance(downsample_kernel_sizes, int):
            downsample_kernel_sizes = [downsample_kernel_sizes] * (depth + 1)
        assert len(downsample_kernel_sizes) == depth + 1
        if isinstance(upsample_kernel_sizes, int):
            upsample_kernel_sizes = [upsample_kernel_sizes] * depth
        assert len(upsample_kernel_sizes) == depth

        # down-sampling
        self._downsample_convs = []
        self._downsample_pools = []
        tensor_shape = image_size
        tensor_shapes = [tensor_shape]
        for d in range(depth):
            downsample_conv = self.build_conv_block(
                filters=num_channels[d],
                kernel_size=downsample_kernel_sizes[d],
                padding=padding,
            )
            downsample_pool = self.build_down_sampling_block(
                kernel_size=strides, strides=strides, padding=padding
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
            self._downsample_convs.append(downsample_conv)
            self._downsample_pools.append(downsample_pool)
            tensor_shapes.append(tensor_shape)

        # bottom layer
        self._bottom_block = self.build_bottom_block(
            filters=num_channels[depth],
            kernel_size=downsample_kernel_sizes[depth],
            padding=padding,
        )

        # up-sampling
        self._upsample_deconvs = []
        self._upsample_convs = []
        for d in range(depth - 1, min_extract_level - 1, -1):
            kernel_size = upsample_kernel_sizes[d]
            output_padding = layer_util.deconv_output_padding(
                input_shape=tensor_shapes[d + 1],
                output_shape=tensor_shapes[d],
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
            )
            upsample_deconv = self.build_up_sampling_block(
                filters=num_channels[d],
                output_padding=output_padding,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                output_shape=tensor_shapes[d],
            )
            upsample_conv = self.build_conv_block(
                filters=num_channels[d], kernel_size=kernel_size, padding=padding
            )
            self._upsample_deconvs = [upsample_deconv] + self._upsample_deconvs
            self._upsample_convs = [upsample_conv] + self._upsample_convs
        if min_extract_level > 0:
            # add Nones to make lists have length depth - 1
            self._upsample_deconvs = [None] * min_extract_level + self._upsample_deconvs
            self._upsample_convs = [None] * min_extract_level + self._upsample_convs

        # extraction
        self._output_block = self.build_output_block(
            image_size=image_size,
            extract_levels=extract_levels,
            out_channels=out_channels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
        )

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
    def build_skip_block(self) -> Union[tf.keras.Model, tfkl.Layer]:
        """
        Build a block for combining skipped tensor and up-sampled one.

        This block do not change the tensor shape (width, height, depth),
        it only changes the number of channels.

        The input to this block is a list of tensors.

        :return: a block consists of one or multiple layers
        """

    @abstractmethod
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

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """
        Build LocalNet graph based on built layers.

        :param inputs: image batch, shape = (batch, f_dim1, f_dim2, f_dim3, ch)
        :param training: None or bool.
        :param mask: None or tf.Tensor.
        :return: shape = (batch, f_dim1, f_dim2, f_dim3, out_channels)
        """

        # down-sampling
        skips = []
        down_sampled = inputs
        for d in range(self._depth):
            skip = self._downsample_convs[d](inputs=down_sampled, training=training)
            down_sampled = self._downsample_pools[d](inputs=skip, training=training)
            skips.append(skip)

        # bottom
        up_sampled = self._bottom_block(inputs=down_sampled, training=training)

        # up-sampling
        outs = [up_sampled]
        for d in range(self._depth - 1, min(self._extract_levels) - 1, -1):
            up_sampled = self._upsample_deconvs[d](inputs=up_sampled, training=training)
            up_sampled = self.build_skip_block()([up_sampled, skips[d]])
            up_sampled = self._upsample_convs[d](inputs=up_sampled, training=training)
            outs.append(up_sampled)

        # output
        output = self._output_block(outs)

        return output


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
        out_channels: int,
        num_channel_initial: int,
        depth: int,
        out_kernel_initializer: str,
        out_activation: str,
        pooling: bool = True,
        concat_skip: bool = False,
        name: str = "Unet",
        **kwargs,
    ):
        """
        Initialise UNet.

        :param image_size: (dim1, dim2, dim3), dims of input image.
        :param out_channels: number of channels for the output
        :param num_channel_initial: number of initial channels
        :param depth: input is at level 0, bottom is at level depth
        :param out_kernel_initializer: kernel initializer for the last layer
        :param out_activation: activation at the last layer
        :param pooling: for downsampling, use non-parameterized
                        pooling if true, otherwise use conv3d
        :param concat_skip: when upsampling, concatenate skipped
                            tensor if true, otherwise use addition
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

        # init layer variables
        num_channels = [num_channel_initial * (2 ** d) for d in range(depth + 1)]

        self._num_channel_initial = num_channel_initial
        self._depth = depth
        self._concat_skip = concat_skip
        self._downsample_convs = []
        self._downsample_pools = []
        tensor_shape = image_size
        self._tensor_shapes = [tensor_shape]
        for d in range(depth):
            downsample_conv = tf.keras.Sequential(
                [
                    layer.Conv3dBlock(
                        filters=num_channels[d], kernel_size=3, padding="same"
                    ),
                    layer.ResidualConv3dBlock(
                        filters=num_channels[d], kernel_size=3, padding="same"
                    ),
                ]
            )
            if pooling:
                downsample_pool = tfkl.MaxPool3D(pool_size=2, strides=2, padding="same")
            else:
                downsample_pool = layer.Conv3dBlock(
                    filters=num_channels[d], kernel_size=3, strides=2, padding="same"
                )
            tensor_shape = tuple((x + 1) // 2 for x in tensor_shape)
            self._downsample_convs.append(downsample_conv)
            self._downsample_pools.append(downsample_pool)
            self._tensor_shapes.append(tensor_shape)
        self._bottom_conv3d = layer.Conv3dBlock(
            filters=num_channels[depth], kernel_size=3, padding="same"
        )
        self._bottom_res3d = layer.ResidualConv3dBlock(
            filters=num_channels[depth], kernel_size=3, padding="same"
        )
        self._upsample_deconvs = []
        self._upsample_convs = []
        for d in range(depth):
            padding = layer_util.deconv_output_padding(
                input_shape=self._tensor_shapes[d + 1],
                output_shape=self._tensor_shapes[d],
                kernel_size=3,
                stride=2,
                padding="same",
            )
            upsample_deconv = layer.Deconv3dBlock(
                filters=num_channels[d],
                output_padding=padding,
                kernel_size=3,
                strides=2,
                padding="same",
            )
            upsample_conv = tf.keras.Sequential(
                [
                    layer.Conv3dBlock(
                        filters=num_channels[d], kernel_size=3, padding="same"
                    ),
                    layer.ResidualConv3dBlock(
                        filters=num_channels[d], kernel_size=3, padding="same"
                    ),
                ]
            )
            self._upsample_deconvs.append(upsample_deconv)
            self._upsample_convs.append(upsample_conv)
        self._output_conv3d = tf.keras.Sequential(
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

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """
        Builds graph based on built layers.

        :param inputs: shape = [batch, f_dim1, f_dim2, f_dim3, in_channels]
        :param training:
        :param mask:
        :return: shape = [batch, f_dim1, f_dim2, f_dim3, out_channels]
        """

        down_sampled = inputs

        # down sample
        skips = []
        for d_var in range(self._depth):  # level 0 to D-1
            skip = self._downsample_convs[d_var](inputs=down_sampled, training=training)
            down_sampled = self._downsample_pools[d_var](inputs=skip, training=training)
            skips.append(skip)

        # bottom, level D
        up_sampled = self._bottom_res3d(
            inputs=self._bottom_conv3d(inputs=down_sampled, training=training),
            training=training,
        )

        # up sample, level D-1 to 0
        for d_var in range(self._depth - 1, -1, -1):
            up_sampled = self._upsample_deconvs[d_var](
                inputs=up_sampled, training=training
            )
            if self._concat_skip:
                up_sampled = tf.concat([up_sampled, skips[d_var]], axis=4)
            else:
                up_sampled = up_sampled + skips[d_var]
            up_sampled = self._upsample_convs[d_var](
                inputs=up_sampled, training=training
            )

        # output
        output = self._output_conv3d(inputs=up_sampled, training=training)

        return output
