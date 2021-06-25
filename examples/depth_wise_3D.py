# https://github.com/alexandrosstergiou/keras-DepthwiseConv3D
# -*- coding: utf-8 -*-

'''
This is a modification of the SeparableConv3D code in Keras,
to perform just the Depthwise Convolution (1st step) of the
Depthwise Separable Convolution layer.
'''
from __future__ import absolute_import

from keras import backend as K
from keras import initializers
from keras import regularizers
from keras import constraints
from keras import layers
from keras.engine import InputSpec
from keras.utils import conv_utils
from keras.legacy.interfaces import conv3d_args_preprocessor, generate_legacy_interface
from keras.layers import Conv3D
from keras.backend.tensorflow_backend import _preprocess_padding, _preprocess_conv3d_input

import tensorflow as tf


def depthwise_conv3d_args_preprocessor(args, kwargs):
    converted = []

    if 'init' in kwargs:
        init = kwargs.pop('init')
        kwargs['depthwise_initializer'] = init
        converted.append(('init', 'depthwise_initializer'))

    args, kwargs, _converted = conv3d_args_preprocessor(args, kwargs)
    return args, kwargs, converted + _converted

    legacy_depthwise_conv3d_support = generate_legacy_interface(
    allowed_positional_args=['filters', 'kernel_size'],
    conversions=[('nb_filter', 'filters'),
                 ('subsample', 'strides'),
                 ('border_mode', 'padding'),
                 ('dim_ordering', 'data_format'),
                 ('b_regularizer', 'bias_regularizer'),
                 ('b_constraint', 'bias_constraint'),
                 ('bias', 'use_bias')],
    value_conversions={'dim_ordering': {'tf': 'channels_last',
                                        'th': 'channels_first',
                                        'default': None}},
    preprocessor=depthwise_conv3d_args_preprocessor)


class DepthwiseConv3D(Conv3D):
    """Depthwise 3D convolution.
    Depth-wise part of separable convolutions consist in performing
    just the first step/operation
    (which acts on each input channel separately).
    It does not perform the pointwise convolution (second step).
    The `depth_multiplier` argument controls how many
    output channels are generated per input channel in the depthwise step.
    # Arguments
        kernel_size: An integer or tuple/list of 3 integers, specifying the
            depth, width and height of the 3D convolution window.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        strides: An integer or tuple/list of 3 integers,
            specifying the strides of the convolution along the depth, width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
        padding: one of `"valid"` or `"same"` (case-insensitive).
        depth_multiplier: The number of depthwise convolution output channels
            for each input channel.
            The total number of depthwise convolution output
            channels will be equal to `filterss_in * depth_multiplier`.
        groups: The depth size of the convolution (as a variant of the original Depthwise conv)
        data_format: A string,
            one of `channels_last` (default) or `channels_first`.
            The ordering of the dimensions in the inputs.
            `channels_last` corresponds to inputs with shape
            `(batch, height, width, channels)` while `channels_first`
            corresponds to inputs with shape
            `(batch, channels, height, width)`.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
        activation: Activation function to use
            (see [activations](../activations.md)).
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        depthwise_initializer: Initializer for the depthwise kernel matrix
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        depthwise_regularizer: Regularizer function applied to
            the depthwise kernel matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        dialation_rate: List of ints.
                        Defines the dilation factor for each dimension in the
                        input. Defaults to (1,1,1)
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        depthwise_constraint: Constraint function applied to
            the depthwise kernel matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
    # Input shape
        5D tensor with shape:
        `(batch, depth, channels, rows, cols)` if data_format='channels_first'
        or 5D tensor with shape:
        `(batch, depth, rows, cols, channels)` if data_format='channels_last'.
    # Output shape
        5D tensor with shape:
        `(batch, filters * depth, new_depth, new_rows, new_cols)` if data_format='channels_first'
        or 4D tensor with shape:
        `(batch, new_depth, new_rows, new_cols, filters * depth)` if data_format='channels_last'.
        `rows` and `cols` values might have changed due to padding.
    """

    #@legacy_depthwise_conv3d_support
    def __init__(self,
                 kernel_size,
                 strides=(1, 1, 1),
                 padding='valid',
                 depth_multiplier=1,
                 groups=None,
                 data_format=None,
                 activation=None,
                 use_bias=True,
                 depthwise_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 dilation_rate = (1, 1, 1),
                 depthwise_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 depthwise_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(DepthwiseConv3D, self).__init__(
            filters=None,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            activation=activation,
            use_bias=use_bias,
            bias_regularizer=bias_regularizer,
            dilation_rate=dilation_rate,
            activity_regularizer=activity_regularizer,
            bias_constraint=bias_constraint,
            **kwargs)
        self.depth_multiplier = depth_multiplier
        self.groups = groups
        self.depthwise_initializer = initializers.get(depthwise_initializer)
        self.depthwise_regularizer = regularizers.get(depthwise_regularizer)
        self.depthwise_constraint = constraints.get(depthwise_constraint)
        self.bias_initializer = initializers.get(bias_initializer)
        self.dilation_rate = dilation_rate
        self._padding = _preprocess_padding(self.padding)
        self._strides = (1,) + self.strides + (1,)
        self._data_format = "NDHWC"
        self.input_dim = None

    def build(self, input_shape):
        if len(input_shape) < 5:
            raise ValueError('Inputs to `DepthwiseConv3D` should have rank 5. '
                             'Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`DepthwiseConv3D` '
                             'should be defined. Found `None`.')
        self.input_dim = int(input_shape[channel_axis])

        if self.groups is None:
            self.groups = self.input_dim

        if self.groups > self.input_dim:
            raise ValueError('The number of groups cannot exceed the number of channels')

        if self.input_dim % self.groups != 0:
            raise ValueError('Warning! The channels dimension is not divisible by the group size chosen')

        depthwise_kernel_shape = (self.kernel_size[0],
                                  self.kernel_size[1],
                                  self.kernel_size[2],
                                  self.input_dim,
                                  self.depth_multiplier)

        self.depthwise_kernel = self.add_weight(
            shape=depthwise_kernel_shape,
            initializer=self.depthwise_initializer,
            name='depthwise_kernel',
            regularizer=self.depthwise_regularizer,
            constraint=self.depthwise_constraint)

        if self.use_bias:
            self.bias = self.add_weight(shape=(self.groups * self.depth_multiplier,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        # Set input spec.
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: self.input_dim})
        self.built = True

    def call(self, inputs, training=None):
        inputs = _preprocess_conv3d_input(inputs, self.data_format)

        if self.data_format == 'channels_last':
            dilation = (1,) + self.dilation_rate + (1,)
        else:
            dilation = self.dilation_rate + (1,) + (1,)

        if self._data_format == 'NCDHW':
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, i:i+self.input_dim//self.groups, :, :, :], self.depthwise_kernel[:, :, :, i:i+self.input_dim//self.groups, :],
                    strides=self._strides,
                    padding=self._padding,
                    dilations=dilation,
                    data_format=self._data_format) for i in range(0, self.input_dim, self.input_dim//self.groups)], axis=1)

        else:
            outputs = tf.concat(
                [tf.nn.conv3d(inputs[0][:, :, :, :, i:i+self.input_dim//self.groups], self.depthwise_kernel[:, :, :, i:i+self.input_dim//self.groups, :],
                    strides=self._strides,
                    padding=self._padding,
                    dilations=dilation,
                    data_format=self._data_format) for i in range(0, self.input_dim, self.input_dim//self.groups)], axis=-1)

        if self.bias is not None:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            depth = input_shape[2]
            rows = input_shape[3]
            cols = input_shape[4]
            out_filters = self.groups * self.depth_multiplier
        elif self.data_format == 'channels_last':
            depth = input_shape[1]
            rows = input_shape[2]
            cols = input_shape[3]
            out_filters = self.groups * self.depth_multiplier

        depth = conv_utils.conv_output_length(depth, self.kernel_size[0],
                                             self.padding,
                                             self.strides[0])

        rows = conv_utils.conv_output_length(rows, self.kernel_size[1],
                                             self.padding,
                                             self.strides[1])

        cols = conv_utils.conv_output_length(cols, self.kernel_size[2],
                                             self.padding,
                                             self.strides[2])

        if self.data_format == 'channels_first':
            return (input_shape[0], out_filters, depth, rows, cols)

        elif self.data_format == 'channels_last':
            return (input_shape[0], depth, rows, cols, out_filters)

    def get_config(self):
        config = super(DepthwiseConv3D, self).get_config()
        config.pop('filters')
        config.pop('kernel_initializer')
        config.pop('kernel_regularizer')
        config.pop('kernel_constraint')
        config['depth_multiplier'] = self.depth_multiplier
        config['depthwise_initializer'] = initializers.serialize(self.depthwise_initializer)
        config['depthwise_regularizer'] = regularizers.serialize(self.depthwise_regularizer)
        config['depthwise_constraint'] = constraints.serialize(self.depthwise_constraint)
        return config

DepthwiseConvolution3D = DepthwiseConv3D