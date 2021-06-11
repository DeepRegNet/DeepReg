"""This script provides an example of using efficient for training."""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from copy import deepcopy
from typing import List, Optional, Tuple, Union

from deepreg.model import layer
from deepreg.model.backbone import Backbone
from deepreg.model.backbone.local_net import LocalNet
from deepreg.model.backbone.u_net import UNet
from deepreg.model.layer import Extraction
from deepreg.registry import REGISTRY
from deepreg.train import train


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

# Each Blocks Parameters
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

# Two Kernel Initializer
CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


@REGISTRY.register_backbone(name="efficient_net")
class EfficientNet(LocalNet):
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
        extract_levels: Tuple[int, ...],
        out_kernel_initializer: str,
        out_activation: str,
        out_channels: int,
        depth: Optional[int] = None,
        pooling: bool = True,
        concat_skip: bool = False,
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
        :param pooling: for down-sampling, use non-parameterized
                        pooling if true, otherwise use conv3d
        :param concat_skip: when up-sampling, concatenate skipped
                            tensor if true, otherwise use addition
        :param width_coefficient: float, scaling coefficient for network width.
        :param depth_coefficient: float, scaling coefficient for network depth.
        :param default_size: int, default input image size.
        :param dropout_rate: float, dropout rate before final classifier layer.
        :param drop_connect_rate: float, dropout rate at skip connections.
        :param depth_divisor: int divisor for depth.
        :param name: name of the backbone.
        :param kwargs: additional arguments.
        """
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
            use_additive_upsampling = False,
            pooling=pooling,
            concat_skip=concat_skip,
            name=name,
            **kwargs,
        )

        self.width_coefficient =  width_coefficient
        self.depth_coefficient = depth_coefficient
        self.default_size = default_size
        self.dropout_rate = dropout_rate
        self.drop_connect_rate = drop_connect_rate
        self.depth_divisor = depth_divisor
        self.activation_fn = tf.nn.swish

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
        decoded = self.build_efficient_net(inputs=encoded, training=training)  # type: ignore

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


    def build_efficient_net(self, inputs: tf.Tensor, training=None) -> tf.Tensor:
        """
        Builds graph based on built layers.

        :param inputs: shape = (batch, f_dim1, f_dim2, f_dim3, in_channels)
        :param training:
        :param mask:
        :return: shape = (batch, f_dim1, f_dim2, f_dim3, out_channels)
        """
        img_input = layers.Input(tensor=inputs, shape=self.image_size)
        bn_axis = 4  
        x = img_input
        # x = layers.ZeroPadding3D(padding=self.correct_pad(x, 3),
        #                         name='stem_conv_pad')(x)

        x = layers.Conv3D(self.round_filters(32), 3,
                        strides=1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='stem_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
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
        
        x = layers.Conv3D(self.round_filters(128), 1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name='top_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name='top_bn')(x)
        x = layers.Activation(self.activation_fn, name='top_activation')(x)

        print("input.shape", inputs.shape, x.shape)
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

    def correct_pad(self, inputs, kernel_size):
        img_dim = 1
        input_size = backend.int_shape(inputs)[img_dim:(img_dim + 3)]

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        if input_size[0] is None:
            adjust = (1, 1, 1)
        else:
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2, 1 - input_size[2] % 2)

        correct = (kernel_size[0] // 2, kernel_size[1] // 2, kernel_size[2] // 2)

        return ((correct[0] - adjust[0], correct[0]),
                (correct[1] - adjust[1], correct[1]),
                (correct[2] - adjust[2], correct[2]))

    def block(self, inputs, activation_fn=tf.nn.swish, drop_rate=0., name='',
            filters_in=32, filters_out=16, kernel_size=3, strides=1,
            expand_ratio=1, se_ratio=0., id_skip=True):

        bn_axis = 4

        filters = filters_in * expand_ratio

        # Inverted residuals
        if expand_ratio != 1:
            x = layers.Conv3D(filters, 1,
                            padding='same',
                            use_bias=False,
                            kernel_initializer=CONV_KERNEL_INITIALIZER,
                            name=name + 'expand_conv')(inputs)
            x = layers.BatchNormalization(axis=bn_axis, name=name + 'expand_bn')(x)
            x = layers.Activation(activation_fn, name=name + 'expand_activation')(x)
        else:
            x = inputs

        # padding
        # if strides == 2:
        #     x = layers.ZeroPadding3D(padding=self.correct_pad(x, kernel_size),
        #                             name=name + 'dwconv_pad')(x)
        #     conv_pad = 'valid'
        # else:
        #     conv_pad = 'same'

        # TODO(Sicong): Find DepthwiseConv3D
        # x = layers.DepthwiseConv2D(kernel_size,
        #                         strides=strides,
        #                         padding=conv_pad,
        #                         use_bias=False,
        #                         depthwise_initializer=CONV_KERNEL_INITIALIZER,
        #                         name=name + 'dwconv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'bn')(x)
        x = layers.Activation(activation_fn, name=name + 'activation')(x)

        if 0 < se_ratio <= 1:
            filters_se = max(1, int(filters_in * se_ratio))
            se = layers.GlobalAveragePooling3D(name=name + 'se_squeeze')(x)
            se = layers.Reshape((1, 1, 1, filters), name=name + 'se_reshape')(se)
            se = layers.Conv3D(filters_se, 1,
                            padding='same',
                            activation=activation_fn,
                            kernel_initializer=CONV_KERNEL_INITIALIZER,
                            name=name + 'se_reduce')(se)
            se = layers.Conv3D(filters, 1,
                            padding='same',
                            activation='sigmoid',
                            kernel_initializer=CONV_KERNEL_INITIALIZER,
                            name=name + 'se_expand')(se)
            x = layers.multiply([x, se], name=name + 'se_excite')

        x = layers.Conv3D(filters_out, 1,
                        padding='same',
                        use_bias=False,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        name=name + 'project_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=name + 'project_bn')(x)

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
        return config


if __name__ == "__main__":
    config_path = "examples/config_efficient_net.yaml"
    train(
        gpu="",
        config_path=config_path,
        gpu_allow_growth=True,
        ckpt_path="",
    )
