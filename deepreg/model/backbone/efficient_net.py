# coding=utf-8

from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfkl
from tensorflow.python.keras.utils import conv_utils

from deepreg.model import layer, layer_util
from deepreg.model.backbone.interface import Backbone
from deepreg.model.layer import Extraction
from deepreg.registry import REGISTRY

EFFICIENTNET_PARAMS = {
    # model_name: (width_mult, depth_mult, image_size, dropout_rate, dropconnect_rate)
    "efficientnet-b0": (1.0, 1.0, 224, 0.2, 0.2),
    "efficientnet-b1": (1.0, 1.1, 240, 0.2, 0.2),
    "efficientnet-b2": (1.1, 1.2, 260, 0.3, 0.2),
    "efficientnet-b3": (1.2, 1.4, 300, 0.3, 0.2),
    "efficientnet-b4": (1.4, 1.8, 380, 0.4, 0.2),
    "efficientnet-b5": (1.6, 2.2, 456, 0.4, 0.2),
    "efficientnet-b6": (1.8, 2.6, 528, 0.5, 0.2),
    "efficientnet-b7": (2.0, 3.1, 600, 0.5, 0.2),
}

DEFAULT_BLOCKS_ARGS = [
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 32, 'filters_out': 16,
     'expand_ratio': 1, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 2, 'filters_in': 16, 'filters_out': 24,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 2, 'filters_in': 24, 'filters_out': 40,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 3, 'filters_in': 40, 'filters_out': 80,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 3, 'filters_in': 80, 'filters_out': 112,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25},
    {'kernel_size': 5, 'repeats': 4, 'filters_in': 112, 'filters_out': 192,
     'expand_ratio': 6, 'id_skip': True, 'strides': 2, 'se_ratio': 0.25},
    {'kernel_size': 3, 'repeats': 1, 'filters_in': 192, 'filters_out': 320,
     'expand_ratio': 6, 'id_skip': True, 'strides': 1, 'se_ratio': 0.25}
]

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


@REGISTRY.register_backbone(name="efficient_net")
class EfficientNet(Backbone):
    """
    Class that implements an Efficient-Net for image registration.

    Reference:
    - Author: Mingxing Tan, Quoc V. Le,
      EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
      https://arxiv.org/pdf/1905.11946.pdf
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
        width_coefficient: float = 1.0,
        depth_coefficient: float = 1.0,
        default_size: int = 224,
        dropout_rate: float = 0.2,
        drop_connect_rate: float = 0.2,
        depth_divisor: int = 8,
        name: str = "EfficientNet",
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
        :param width_coefficient: float, scaling coefficient for network width.
        :param depth_coefficient: float, scaling coefficient for network depth.
        :param default_size: int, default input image size.
        :param dropout_rate: float, dropout rate before final classifier layer.
        :param drop_connect_rate: float, dropout rate at skip connections.
        :param depth_divisor: int divisor for depth.
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

        # efficient parameters
        self._width_coefficient =  width_coefficient
        self._depth_coefficient = depth_coefficient
        self._default_size = default_size
        self._dropout_rate = dropout_rate
        self._drop_connect_rate = drop_connect_rate
        self._depth_divisor = depth_divisor
        self._activation_fn = tf.nn.swish

        # init layers
        # all lists start with d = 0
        self._encode_convs: List[tfkl.Layer] = []
        self._encode_pools: List[tfkl.Layer] = []
        self._bottom_block = None
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
        tensor_shapes = self.build_encode_layers(
            image_size=image_size,
            num_channels=encode_num_channels,
            depth=depth,
            encode_kernel_sizes=encode_kernel_sizes,
            strides=strides,
            padding=padding,
        )
        self._output_block = self.build_output_block(
            image_size=image_size,
            extract_levels=extract_levels,
            out_channels=out_channels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
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

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """
        Build compute graph based on built layers.

        :param inputs: image batch, shape = (batch, f_dim1, f_dim2, f_dim3, ch)
        :param training: None or bool.
        :param mask: None or tf.Tensor.
        :return: shape = (batch, f_dim1, f_dim2, f_dim3, out_channels)
        """

        # encoding / down-sampling
        # skips = []
        # encoded = inputs
        # for d in range(self._depth):
        #     skip = self._encode_convs[d](inputs=encoded, training=training)
        #     encoded = self._encode_pools[d](inputs=skip, training=training)
        #     skips.append(skip)

        # bottom
        # decoded = self._bottom_block(inputs=encoded, training=training)  # type: ignore

        # decoding / up-sampling. TODO(SicongLu): Add efficient_net based decoder. 

        # output
        decoded = self.build_efficient_net(inputs=encoded)  # type: ignore
        outs = [decoded]
        output = self._output_block(outs)  # type: ignore

        return output

    def build_efficient_net(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        """
        Builds graph based on built layers.

        :param inputs: shape = (batch, f_dim1, f_dim2, f_dim3, in_channels)
        :param training:
        :param mask:
        :return: shape = (batch, f_dim1, f_dim2, f_dim3, out_channels)
        """
        x = inputs
        x = layers.Conv3D(32, 3,
                        strides=1,
                        padding='same',
                        use_bias=False,
                        # kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='stem_conv')(x)
        x = layers.BatchNormalization(axis=4, name='stem_bn')(x)
        x = layers.Activation(self.activation_fn, name='stem_activation')(x)
        blocks_args = deepcopy(DEFAULT_BLOCKS_ARGS)

        b = 0
        # Calculate the number of blocks
        blocks = float(sum(args['repeats'] for args in blocks_args))
        for (i, args) in enumerate(blocks_args):
            assert args['repeats'] > 0
            args['filters_in'] = self.round_filters(args['filters_in'])
            args['filters_out'] = self.round_filters(args['filters_out'])

            for j in range(self.round_repeats(args.pop('repeats'))):
                if j > 0:
                    args['strides'] = 1
                    args['filters_in'] = args['filters_out']
                x = self.block(x, self.activation_fn, self.drop_connect_rate * b / blocks,
                        name='block{}{}_'.format(i + 1, chr(j + 97)), **args)
                b += 1
        
        x = layers.Conv3D(128, 1,
                        padding='same',
                        use_bias=False,
                        name='top_conv')(x)
        x = layers.BatchNormalization(axis=4, name='top_bn')(x)
        x = layers.Activation(self.activation_fn, name='top_activation')(x)

        return x

    def round_filters(self, filters):
        """Round number of filters based on depth multiplier."""
        filters *= self.width_coefficient
        divisor = self.depth_divisor
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_filters < 0.9 * filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(self, repeats):
        return int(math.ceil(self.depth_coefficient * repeats))

    def block(self, inputs, activation_fn=tf.nn.swish, drop_rate=0., name='',
            filters_in=32, filters_out=16, kernel_size=3, strides=1,
            expand_ratio=1, se_ratio=0., id_skip=True):
        filters = filters_in * expand_ratio

        # Inverted residuals
        if expand_ratio != 1:
            x = layers.Conv3D(filters, 1,
                            padding='same',
                            use_bias=False,
                            name=name + 'expand_conv')(inputs)
            x = layers.BatchNormalization(axis=4, name=name + 'expand_bn')(x)
            x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
        else:
            x = inputs

        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            se = layers.GlobalAveragePooling3D(name=name + 'se_squeeze')(x)
            se = layers.Reshape((1, 1, 1, filters), name=name + 'se_reshape')(se)
            se = layers.Conv3D(filters_se, 1,
                            padding='same',
                            activation=activation_fn,
                            name=name + 'se_reduce')(se)
            se = layers.Conv3D(filters, 1,
                            padding='same',
                            activation='sigmoid',
                            name=name + 'se_expand')(se)
            x = layers.multiply([x, se], name=name + 'se_excite')

        x = layers.Conv3D(filters_out, 1,
                        padding='same',
                        use_bias=False,
                        name=name + 'project_conv')(x)
        x = layers.BatchNormalization(axis=4, name=name + 'project_bn')(x)

        if (id_skip is True and strides == 1 and filters_in == filters_out):
            if drop_rate > 0:
                x = layers.Dropout(drop_rate,
                                noise_shape=None,
                                name=name + 'drop')(x)
            x = layers.add([x, inputs], name=name + 'add')

        return x



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
