import itertools

import numpy as np
import tensorflow as tf

import deepreg.model.layer_util as layer_util


class Activation(tf.keras.layers.Layer):
    def __init__(self, identifier: str = "relu", **kwargs):
        """
        Layer wraps tf.keras.activations.get().

        :param identifier: e.g. "relu"
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        self._act = tf.keras.activations.get(identifier=identifier)

    def call(self, inputs, **kwargs):
        return self._act(inputs)


class Norm(tf.keras.layers.Layer):
    def __init__(self, name: str = "batch_norm", axis: int = -1, **kwargs):
        """
        Class merges batch norm and layer norm.

        :param name: str, batch_norm or layer_norm
        :param axis: int
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        if name == "batch_norm":
            self._norm = tf.keras.layers.BatchNormalization(axis=axis, **kwargs)
        elif name == "layer_norm":
            self._norm = tf.keras.layers.LayerNormalization(axis=axis)
        else:
            raise ValueError("Unknown normalization type")

    def call(self, inputs, training=None, **kwargs):
        return self._norm(inputs=inputs, training=training)


class MaxPool3d(tf.keras.layers.Layer):
    def __init__(
        self,
        pool_size: (int, tuple),
        strides: (int, tuple, None) = None,
        padding: str = "same",
        **kwargs,
    ):
        """
        Layer wraps tf.keras.layers.MaxPool3D

        :param pool_size: int or tuple of 3 ints
        :param strides: int or tuple of 3 ints, if None default will be pool_size
        :param padding: str, same or valid
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        self._max_pool = tf.keras.layers.MaxPool3D(
            pool_size=pool_size, strides=strides, padding=padding
        )

    def call(self, inputs, **kwargs):
        return self._max_pool(inputs=inputs)


class Conv3d(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        activation: (str, None) = None,
        use_bias: bool = True,
        kernel_initializer: str = "glorot_uniform",
        **kwargs,
    ):
        """
        Layer wraps tf.keras.layers.Conv3D.

        :param filters: number of channels of the output
        :param kernel_size: int or tuple of 3 ints, e.g. (3,3,3) or 3
        :param strides: int or tuple of 3 ints, e.g. (1,1,1) or 1
        :param padding: same or valid
        :param activation: defines the activation function
        :param use_bias: whether add bias to output
        :param kernel_initializer: defines the initialization method,
            defines the initialization method
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        self._conv3d = tf.keras.layers.Conv3D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
        )

    def call(self, inputs, **kwargs):
        return self._conv3d(inputs=inputs)


class Deconv3d(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        output_shape: (tuple, None) = None,
        kernel_size: int = 3,
        strides: int = 1,
        padding: str = "same",
        use_bias: bool = True,
        **kwargs,
    ):
        """
        Layer wraps tf.keras.layers.Conv3DTranspose
        and does not requires input shape when initializing.

        :param filters: number of channels of the output
        :param output_shape: (out_dim1, out_dim2, out_dim3)
        :param kernel_size: int or tuple of 3 ints, e.g. (3,3,3) or 3
        :param strides: int or tuple of 3 ints, e.g. (1,1,1) or 1
        :param padding: same or valid.
        :param use_bias: use bias for Conv3DTranspose or not.
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # save parameters
        self._filters = filters
        self._output_shape = output_shape
        self._kernel_size = kernel_size
        self._strides = strides
        self._padding = padding
        self._use_bias = use_bias
        self._kwargs = kwargs
        # init layer variables
        self._output_padding = None
        self._deconv3d = None

    def build(self, input_shape):
        super().build(input_shape)

        if isinstance(self._kernel_size, int):
            self._kernel_size = [self._kernel_size] * 3
        if isinstance(self._strides, int):
            self._strides = [self._strides] * 3

        if self._output_shape is not None:
            # pylint: disable-next=line-too-long
            """
            https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/utils/conv_utils.py#L139-L185
            When the output shape is defined, the padding should be calculated manually
            if padding == 'same':
                pad = filter_size // 2
                length = ((input_length - 1) * stride + filter_size
                         - 2 * pad + output_padding)
            """
            self._padding = "same"
            self._output_padding = [
                self._output_shape[i]
                - (
                    (input_shape[1 + i] - 1) * self._strides[i]
                    + self._kernel_size[i]
                    - 2 * (self._kernel_size[i] // 2)
                )
                for i in range(3)
            ]
        self._deconv3d = tf.keras.layers.Conv3DTranspose(
            filters=self._filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            padding=self._padding,
            output_padding=self._output_padding,
            use_bias=self._use_bias,
            **self._kwargs,
        )

    def call(self, inputs, **kwargs):
        return self._deconv3d(inputs=inputs)


class Conv3dBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: (int, tuple) = 3,
        strides: (int, tuple) = 1,
        padding: str = "same",
        **kwargs,
    ):
        """
        A conv3d block having conv3d - norm - activation.

        :param filters: number of channels of the output
        :param kernel_size: int or tuple of 3 ints, e.g. (3,3,3) or 3
        :param strides: int or tuple of 3 ints, e.g. (1,1,1) or 1
        :param padding: str, same or valid
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # init layer variables
        self._conv3d = Conv3d(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
        )
        self._norm = Norm()
        self._act = Activation()

    def call(self, inputs, training=None, **kwargs) -> tf.Tensor:
        """
        :param inputs: shape = (batch, in_dim1, in_dim2, in_dim3, channels)
        :param training: training flag for normalization layers (default: None)
        :param kwargs: additional arguments.
        :return: shape = (batch, in_dim1, in_dim2, in_dim3, channels)
        """
        output = self._conv3d(inputs=inputs)
        output = self._norm(inputs=output, training=training)
        output = self._act(output)
        return output


class Deconv3dBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        output_shape: (tuple, None) = None,
        kernel_size: (int, tuple) = 3,
        strides: (int, tuple) = 1,
        padding: str = "same",
        **kwargs,
    ):
        """
        A deconv3d block having deconv3d - norm - activation.

        :param filters: number of channels of the output
        :param output_shape: (out_dim1, out_dim2, out_dim3)
        :param kernel_size: int or tuple of 3 ints, e.g. (3,3,3) or 3
        :param strides: int or tuple of 3 ints, e.g. (1,1,1) or 1
        :param padding: str, same or valid
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # init layer variables
        self._deconv3d = Deconv3d(
            filters=filters,
            output_shape=output_shape,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=False,
        )
        self._norm = Norm()
        self._act = Activation()

    def call(self, inputs, training=None, **kwargs) -> tf.Tensor:
        """
        :param inputs: shape = (batch, in_dim1, in_dim2, in_dim3, channels)
        :param training: training flag for normalization layers (default: None)
        :param kwargs: additional arguments.
        :return output: shape = (batch, in_dim1, in_dim2, in_dim3, channels)
        """
        output = self._deconv3d(inputs=inputs)
        output = self._norm(inputs=output, training=training)
        output = self._act(output)
        return output


class Residual3dBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: (int, tuple) = 3,
        strides: (int, tuple) = 1,
        **kwargs,
    ):
        """
        A resnet conv3d block.

        1. conved = conv3d(conv3d_block(inputs))
        2. out = act(norm(conved) + inputs)

        :param filters: int, number of filters in the convolutional layers
        :param kernel_size: int or tuple of 3 ints, e.g. (3,3,3) or 3
        :param strides: int or tuple of 3 ints, e.g. (1,1,1) or 1
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # init layer variables
        self._conv3d_block = Conv3dBlock(
            filters=filters, kernel_size=kernel_size, strides=strides
        )
        self._conv3d = Conv3d(
            filters=filters, kernel_size=kernel_size, strides=strides, use_bias=False
        )
        self._norm = Norm()
        self._act = Activation()

    def call(self, inputs, training=None, **kwargs) -> tf.Tensor:
        """
        :param inputs: shape = (batch, in_dim1, in_dim2, in_dim3, channels)
        :param training: training flag for normalization layers (default: None)
        :param kwargs: additional arguments.
        :return output: shape = (batch, in_dim1, in_dim2, in_dim3, channels)
        """
        return self._act(
            self._norm(
                inputs=self._conv3d(inputs=self._conv3d_block(inputs)),
                training=training,
            )
            + inputs
        )


class DownSampleResnetBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: (int, tuple) = 3,
        pooling: bool = True,
        **kwargs,
    ):
        """
        A down-sampling resnet conv3d block, with max-pooling or conv3d.

        1. conved = conv3d_block(inputs)  # adjust channel
        2. skip = residual_block(conved)  # develop feature
        3. pooled = pool(skip) # down-sample

        :param filters: number of channels of the output
        :param kernel_size: int or tuple of 3 ints, e.g. (3,3,3) or 3
        :param pooling: if True, use max pooling to downsample, otherwise use conv.
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # save parameters
        self._pooling = pooling
        # init layer variables
        self._conv3d_block = Conv3dBlock(filters=filters, kernel_size=kernel_size)
        self._residual_block = Residual3dBlock(filters=filters, kernel_size=kernel_size)
        self._max_pool3d = (
            MaxPool3d(pool_size=(2, 2, 2), strides=(2, 2, 2)) if pooling else None
        )
        self._conv3d_block3 = (
            None
            if pooling
            else Conv3dBlock(filters=filters, kernel_size=kernel_size, strides=2)
        )

    def call(self, inputs, training=None, **kwargs) -> (tf.Tensor, tf.Tensor):
        """
        :param inputs: shape = (batch, in_dim1, in_dim2, in_dim3, channels)
        :param training: training flag for normalization layers (default: None)
        :param kwargs: additional arguments.
        :return: (pooled, skip)

          - downsampled, shape = (batch, in_dim1//2, in_dim2//2, in_dim3//2, channels)
          - skipped, shape = (batch, in_dim1, in_dim2, in_dim3, channels)
        """
        conved = self._conv3d_block(inputs=inputs, training=training)  # adjust channel
        skip = self._residual_block(inputs=conved, training=training)  # develop feature
        pooled = (
            self._max_pool3d(inputs=skip)
            if self._pooling
            else self._conv3d_block3(inputs=skip, training=training)
        )  # downsample
        return pooled, skip


class UpSampleResnetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, concat=False, **kwargs):
        """
        An up-sampling resnet conv3d block, with deconv3d.

        :param filters: number of channels of the output
        :param kernel_size: int or tuple of 3 ints, e.g. (3,3,3) or 3
        :param concat: bool,specify how to combine input and skip connection images.
            If True, use concatenation, otherwise use sum (default=False).
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # save parameters
        self._filters = filters
        self._concat = concat
        # init layer variables
        self._deconv3d_block = None
        self._conv3d_block = Conv3dBlock(filters=filters, kernel_size=kernel_size)
        self._residual_block = Residual3dBlock(filters=filters, kernel_size=kernel_size)

    def build(self, input_shape):
        """
        :param input_shape: tuple, (downsampled_image_shape, skip_image_shape)
        """
        super().build(input_shape)
        skip_shape = input_shape[1][1:4]
        self._deconv3d_block = Deconv3dBlock(
            filters=self._filters, output_shape=skip_shape, strides=2
        )

    def call(self, inputs, training=None, **kwargs) -> tf.Tensor:
        r"""
        :param inputs: tuple

          - down-sampled
          - skipped

        :param training: training flag for normalization layers (default: None)
        :param kwargs: additional arguments.
        :return: shape = (batch, \*skip_connection_image_shape, filters]
        """
        up_sampled, skip = inputs[0], inputs[1]
        up_sampled = self._deconv3d_block(
            inputs=up_sampled, training=training
        )  # up sample and change channel
        up_sampled = (
            tf.concat([up_sampled, skip], axis=4) if self._concat else up_sampled + skip
        )  # combine
        up_sampled = self._conv3d_block(
            inputs=up_sampled, training=training
        )  # adjust channel
        up_sampled = self._residual_block(inputs=up_sampled, training=training)  # conv
        return up_sampled


class Conv3dWithResize(tf.keras.layers.Layer):
    def __init__(
        self,
        output_shape: tuple,
        filters: int,
        kernel_initializer: str = "glorot_uniform",
        activation: (str, None) = None,
        **kwargs,
    ):
        """
        A layer contains conv3d - resize3d.

        :param output_shape: tuple, (out_dim1, out_dim2, out_dim3)
        :param filters: int, number of channels of the output
        :param kernel_initializer: str, defines the initialization method
        :param activation: str, defines the activation function
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # save parameters
        self._output_shape = output_shape
        # init layer variables
        self._conv3d = Conv3d(
            filters=filters,
            strides=1,
            kernel_initializer=kernel_initializer,
            activation=activation,
        )  # if not zero, with init NN, ddf may be too large

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """
        :param inputs: shape = (batch, dim1, dim2, dim3, channels)
        :param kwargs: additional arguments.
        :return: shape = (batch, out_dim1, out_dim2, out_dim3, channels)
        """
        output = self._conv3d(inputs=inputs)
        output = layer_util.resize3d(image=output, size=self._output_shape)
        return output


class Warping(tf.keras.layers.Layer):
    def __init__(self, fixed_image_size: tuple, **kwargs):
        """
        A layer warps an image using DDF.

        Reference:

        - transform of neuron
          https://github.com/adalca/neurite/blob/legacy/neuron/utils.py

          where vol = image, loc_shift = ddf

        :param fixed_image_size: shape = (f_dim1, f_dim2, f_dim3)
             or (f_dim1, f_dim2, f_dim3, ch) with the last channel for features
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        self.grid_ref = tf.expand_dims(
            layer_util.get_reference_grid(grid_size=fixed_image_size), axis=0
        )  # shape = (1, f_dim1, f_dim2, f_dim3, 3)

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """
        :param inputs: (ddf, image)

          - ddf, shape = (batch, f_dim1, f_dim2, f_dim3, 3), dtype = float32
          - image, shape = (batch, m_dim1, m_dim2, m_dim3), dtype = float32
        :param kwargs: additional arguments.
        :return: shape = (batch, f_dim1, f_dim2, f_dim3)
        """
        return layer_util.warp_image_ddf(
            image=inputs[1], ddf=inputs[0], grid_ref=self.grid_ref
        )


class IntDVF(tf.keras.layers.Layer):
    def __init__(self, fixed_image_size: tuple, num_steps: int = 7, **kwargs):
        """
        Layer calculates DVF from DDF.

        Reference:

        - integrate_vec of neuron
          https://github.com/adalca/neurite/blob/legacy/neuron/utils.py

        :param fixed_image_size: tuple, (f_dim1, f_dim2, f_dim3)
        :param num_steps: int, number of steps for integration
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        self._warping = Warping(fixed_image_size=fixed_image_size)
        self._num_steps = num_steps

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """
        :param inputs: dvf, shape = (batch, f_dim1, f_dim2, f_dim3, 3), type = float32
        :param kwargs: additional arguments.
        :return: ddf, shape = (batch, f_dim1, f_dim2, f_dim3, 3)
        """
        ddf = inputs / (2 ** self._num_steps)
        for _ in range(self._num_steps):
            ddf += self._warping(inputs=[ddf, ddf])
        return ddf


class Dense(tf.keras.layers.Layer):
    def __init__(self, units: int, bias_initializer: str = "zeros", **kwargs):
        """
        Layer wraps tf.keras.layers.Dense and flattens input if necessary.

        :param units: number of hidden units
        :param bias_initializer: str, default "zeros"
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)

        # init layer variables
        self._flatten = tf.keras.layers.Flatten()
        self._dense = tf.keras.layers.Dense(
            units=units, bias_initializer=bias_initializer
        )

    def call(self, inputs, **kwargs) -> tf.Tensor:
        r"""
        :param inputs: shape = (batch, \*vol_dim, channels)
        :param kwargs: (not used)
        :return: shape = (batch, units)
        """
        flatten_inputs = self._flatten(inputs)
        return self._dense(flatten_inputs)


class AdditiveUpSampling(tf.keras.layers.Layer):
    def __init__(self, output_shape: tuple, stride: (int, list) = 2, **kwargs):
        """
        Layer up-samples 3d tensor and reduce channels using split and sum.

        :param output_shape: (out_dim1, out_dim2, out_dim3)
        :param stride: int, 1-D Tensor or list
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # save parameters
        self._stride = stride
        self._output_shape = output_shape

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """
        :param inputs: shape = (batch, dim1, dim2, dim3, channels)
        :param kwargs: additional arguments.
        :return: shape = (batch, out_dim1, out_dim2, out_dim3, channels//stride]
        """
        if inputs.shape[4] % self._stride != 0:
            raise ValueError("The channel dimension can not be divided by the stride")
        output = layer_util.resize3d(image=inputs, size=self._output_shape)
        # a list of (batch, out_dim1, out_dim2, out_dim3, channels//stride)
        output = tf.split(output, num_or_size_splits=self._stride, axis=4)
        # (batch, out_dim1, out_dim2, out_dim3, channels//stride)
        output = tf.reduce_sum(tf.stack(output, axis=5), axis=5)
        return output


class LocalNetResidual3dBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: (int, tuple) = 3,
        strides: (int, tuple) = 1,
        **kwargs,
    ):
        """
        A resnet conv3d block, simpler than Residual3dBlock.

        1. conved = conv3d(inputs)
        2. out = act(norm(conved) + inputs)

        :param filters: number of channels of the output
        :param kernel_size: int or tuple of 3 ints, e.g. (3,3,3) or 3
        :param strides: int or tuple of 3 ints, e.g. (1,1,1) or 1
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # init layer variables
        self._conv3d = Conv3d(
            filters=filters, kernel_size=kernel_size, strides=strides, use_bias=False
        )
        self._norm = Norm()
        self._act = Activation()

    def call(self, inputs, training=None, **kwargs) -> tf.Tensor:
        return self._act(
            self._norm(inputs=self._conv3d(inputs=inputs[0]), training=training)
            + inputs[1]
        )


class LocalNetUpSampleResnetBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, use_additive_upsampling: bool = True, **kwargs):
        """
        Layer up-samples tensor with two inputs (skipped and down-sampled).

        :param filters: int, number of output channels
        :param use_additive_upsampling: bool to used additive upsampling
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        # save parameters
        self._filters = filters
        self._use_additive_upsampling = use_additive_upsampling
        # init layer variables
        self._deconv3d_block = None
        self._additive_upsampling = None
        self._conv3d_block = Conv3dBlock(filters=filters)
        self._residual_block = LocalNetResidual3dBlock(filters=filters, strides=1)

    def build(self, input_shape):
        """
        :param input_shape: tuple (nonskip_tensor_shape, skip_tensor_shape)
        """
        super().build(input_shape)

        output_shape = input_shape[1][1:4]
        self._deconv3d_block = Deconv3dBlock(
            filters=self._filters, output_shape=output_shape, strides=2
        )
        if self._use_additive_upsampling:
            self._additive_upsampling = AdditiveUpSampling(output_shape=output_shape)

    def call(self, inputs, training=None, **kwargs) -> tf.Tensor:
        """
        :param inputs: list = [inputs_nonskip, inputs_skip]
        :param training: training flag for normalization layers (default: None)
        :param kwargs: additional arguments.
        :return:
        """
        inputs_nonskip, inputs_skip = inputs[0], inputs[1]
        h0 = self._deconv3d_block(inputs=inputs_nonskip, training=training)
        if self._use_additive_upsampling:
            h0 += self._additive_upsampling(inputs=inputs_nonskip)
        r1 = h0 + inputs_skip
        r2 = self._conv3d_block(inputs=h0, training=training)
        h1 = self._residual_block(inputs=[r2, r1], training=training)
        return h1


class ResizeCPTransform(tf.keras.layers.Layer):
    """
    Layer for getting the control points from the output of a image-to-image network.
    It uses an anti-aliasing Gaussian filter before downsampling.
    """

    def __init__(self, control_point_spacing: (list, tuple, int), **kwargs):
        """
        :param control_point_spacing: list or int
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)

        if isinstance(control_point_spacing, int):
            control_point_spacing = [control_point_spacing] * 3

        self.kernel_sigma = [
            0.44 * cp for cp in control_point_spacing
        ]  # 0.44 = ln(4)/pi
        self.cp_spacing = control_point_spacing
        self.kernel = None
        self._output_shape = None

    def build(self, input_shape):
        super().build(input_shape=input_shape)

        self.kernel = layer_util.gaussian_filter_3d(self.kernel_sigma)
        output_shape = [
            tf.cast(tf.math.ceil(v / c) + 3, tf.int32)
            for v, c in zip(input_shape[1:-1], self.cp_spacing)
        ]
        self._output_shape = output_shape

    def call(self, inputs, **kwargs) -> tf.Tensor:
        output = tf.nn.conv3d(
            inputs, self.kernel, strides=(1, 1, 1, 1, 1), padding="SAME"
        )
        return layer_util.resize3d(image=output, size=self._output_shape)


class BSplines3DTransform(tf.keras.layers.Layer):
    """
     Layer for BSplines interpolation with precomputed cubic spline filters.
     It assumes a full sized image from which:
     1. it compute the contol points values by downsampling the initial image
     2. performs the interpolation
     3. crops the image around the valid values.

    :param cp_spacing: int or tuple of three ints specifying the spacing (in pixels)
        in each dimension. When a single int is used,
        the same spacing to all dimensions is used
    :param output_shape: (batch_size, dim0, dim1, dim2, 3) of the high resolution
        deformation fields.
    :param kwargs: additional arguments.
    """

    def __init__(self, cp_spacing: (int, tuple), output_shape: tuple, **kwargs):

        super().__init__(**kwargs)

        self.filters = []
        self._output_shape = output_shape

        if isinstance(cp_spacing, int):
            self.cp_spacing = (cp_spacing, cp_spacing, cp_spacing)
        else:
            self.cp_spacing = cp_spacing

    def build(self, input_shape: tuple):
        """
        :param input_shape: tuple with the input shape
        :return: None
        """

        super().build(input_shape=input_shape)

        b = {
            0: lambda u: np.float64((1 - u) ** 3 / 6),
            1: lambda u: np.float64((3 * (u ** 3) - 6 * (u ** 2) + 4) / 6),
            2: lambda u: np.float64((-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6),
            3: lambda u: np.float64(u ** 3 / 6),
        }

        filters = np.zeros(
            (
                4 * self.cp_spacing[0],
                4 * self.cp_spacing[1],
                4 * self.cp_spacing[2],
                3,
                3,
            ),
            dtype=np.float32,
        )

        u_arange = 1 - np.arange(
            1 / (2 * self.cp_spacing[0]), 1, 1 / self.cp_spacing[0]
        )
        v_arange = 1 - np.arange(
            1 / (2 * self.cp_spacing[1]), 1, 1 / self.cp_spacing[1]
        )
        w_arange = 1 - np.arange(
            1 / (2 * self.cp_spacing[2]), 1, 1 / self.cp_spacing[2]
        )

        filter_idx = [[0, 1, 2, 3] for _ in range(3)]
        filter_coord = list(itertools.product(*filter_idx))

        for f_idx in filter_coord:
            for it_dim in range(3):
                filters[
                    f_idx[0] * self.cp_spacing[0] : (f_idx[0] + 1) * self.cp_spacing[0],
                    f_idx[1] * self.cp_spacing[1] : (f_idx[1] + 1) * self.cp_spacing[1],
                    f_idx[2] * self.cp_spacing[2] : (f_idx[2] + 1) * self.cp_spacing[2],
                    it_dim,
                    it_dim,
                ] = (
                    b[f_idx[0]](u_arange)[:, None, None]
                    * b[f_idx[1]](v_arange)[None, :, None]
                    * b[f_idx[2]](w_arange)[None, None, :]
                )

        self.filter = tf.convert_to_tensor(filters)

    def interpolate(self, field) -> tf.Tensor:
        """
        :param field: tf.Tensor with shape=number_of_control_points_per_dim
        :return: interpolated_field: tf.Tensor
        """

        image_shape = tuple(
            [(a - 1) * b + 4 * b for a, b in zip(field.shape[1:-1], self.cp_spacing)]
        )

        output_shape = (field.shape[0],) + image_shape + (3,)
        return tf.nn.conv3d_transpose(
            field,
            self.filter,
            output_shape=output_shape,
            strides=self.cp_spacing,
            padding="VALID",
        )

    def call(self, inputs, **kwargs) -> tf.Tensor:
        """
        :param inputs: tf.Tensor defining a low resolution free-form deformation field
        :param kwargs: additional arguments.
        :return: interpolated_field: tf.Tensor of shape=self.input_shape
        """
        high_res_field = self.interpolate(inputs)

        index = [int(3 * c) for c in self.cp_spacing]
        return high_res_field[
            :,
            index[0] : index[0] + self._output_shape[0],
            index[1] : index[1] + self._output_shape[1],
            index[2] : index[2] + self._output_shape[2],
        ]
