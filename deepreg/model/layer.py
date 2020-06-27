import tensorflow as tf

import deepreg.model.layer_util as layer_util


class Activation(tf.keras.layers.Layer):
    def __init__(self, identifier="relu", **kwargs):
        """
        :param identifier: e.g. "relu"
        :param kwargs:
        """
        super(Activation, self).__init__(**kwargs)
        self._act = tf.keras.activations.get(identifier=identifier)

    def call(self, inputs, **kwargs):
        return self._act(inputs)


class Norm(tf.keras.layers.Layer):
    def __init__(self, name="batch_norm", axis=-1, **kwargs):
        super(Norm, self).__init__(**kwargs)
        if name == "batch_norm":
            self._norm = tf.keras.layers.BatchNormalization(axis=axis, **kwargs)
        elif name == "layer_norm":
            self._norm = tf.keras.layers.LayerNormalization(axis=axis)
        else:
            raise ValueError("Unknown normalization type")

    def call(self, inputs, training=None, **kwargs):
        return self._norm(inputs=inputs, training=training)


class MaxPool3d(tf.keras.layers.Layer):
    def __init__(self, pool_size, strides=None, padding="same", **kwargs):
        super(MaxPool3d, self).__init__(**kwargs)
        self._max_pool = tf.keras.layers.MaxPool3D(
            pool_size=pool_size, strides=strides, padding=padding
        )

    def call(self, inputs, **kwargs):
        return self._max_pool(inputs=inputs)


class Conv3d(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size=3,
        strides=1,
        padding="same",
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        **kwargs,
    ):
        """
        :param filters: number of channels of the output
        :param kernel_size: e.g. (3,3,3) or 3
        :param strides: e.g. (1,1,1) or 1
        :param padding: "valid" or "same"
        :param activation:
        :param use_bias:
        :param kernel_initializer:
        """
        super(Conv3d, self).__init__(**kwargs)
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
        filters,
        output_shape=None,
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=True,
        **kwargs,
    ):
        """
        :param filters:
        :param output_shape: [out_dim1, out_dim2, out_dim3]
        :param kernel_size:
        :param strides:
        :param padding:
        :param kwargs:
        """
        super(Deconv3d, self).__init__(**kwargs)
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
        self._Conv3DTranspose = None

    def build(self, input_shape):
        super(Deconv3d, self).build(input_shape)

        if isinstance(self._kernel_size, int):
            self._kernel_size = [self._kernel_size] * 3
        if isinstance(self._strides, int):
            self._strides = [self._strides] * 3

        if self._output_shape is not None:
            """
            https://github.com/tensorflow/tensorflow/blob/1cf0898dd4331baf93fe77205550f2c2e6c90ee5/tensorflow/python/keras/utils/conv_utils.py#L139-L185
            When the output shape is defined, the padding should be calculated manually
            if padding == 'same':
                pad = filter_size // 2
                length = ((input_length - 1) * stride + filter_size - 2 * pad + output_padding)
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
        self._Conv3DTranspose = tf.keras.layers.Conv3DTranspose(
            filters=self._filters,
            kernel_size=self._kernel_size,
            strides=self._strides,
            padding=self._padding,
            output_padding=self._output_padding,
            use_bias=self._use_bias,
            **self._kwargs,
        )

    def call(self, inputs, **kwargs):
        return self._Conv3DTranspose(inputs=inputs)


class Resize3d(tf.keras.layers.Layer):
    def __init__(self, size, method=tf.image.ResizeMethod.BILINEAR, **kwargs):
        """
        :param size: [out_dim1, out_dim2, out_dim3], list or tuple
        :param method:
        :param kwargs:
        """
        super(Resize3d, self).__init__(**kwargs)
        # save parameters
        self._size = size
        self._method = method

    def call(self, inputs, **kwargs):
        """
        tensorflow does not have resize 3d, therefore the resize is performed two folds.
        - resize dim2 and dim3
        - resize dim1 and dim2
        :param inputs: shape = [batch, dim1, dim2, dim3, channels], assuming channels_last
        :return: shape = [batch, out_dim1, out_dim2, out_dim3, channels]
        """

        input_shape = inputs.shape
        # merge axis 0 and 1
        output = tf.reshape(
            inputs, [-1, input_shape[2], input_shape[3], input_shape[4]]
        )  # [batch * dim1, dim2, dim3, channels]
        # resize dim2 and dim3
        output = tf.image.resize(
            images=output, size=[self._size[1], self._size[2]], method=self._method
        )  # [batch * dim1, out_dim2, out_dim3, channels]

        # split axis 0 and merge axis 3 and 4
        output = tf.reshape(
            output,
            shape=[-1, input_shape[1], self._size[1], self._size[2] * input_shape[4]],
        )  # [batch, dim1, out_dim2, out_dim3 * channels]
        # resize dim1 and dim2
        output = tf.image.resize(
            images=output, size=[self._size[0], self._size[1]], method=self._method
        )  # [batch, out_dim1, out_dim2, out_dim3 * channels]
        # reshape
        output = tf.reshape(
            output,
            shape=[-1, self._size[0], self._size[1], self._size[2], input_shape[4]],
        )  # [batch, out_dim1, out_dim2, out_dim3, channels]
        return output


class Conv3dBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, padding="same", **kwargs):
        super(Conv3dBlock, self).__init__(**kwargs)
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

    def call(self, inputs, training=None, **kwargs):
        output = self._conv3d(inputs=inputs)
        output = self._norm(inputs=output, training=training)
        output = self._act(output)
        return output


class Deconv3dBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        output_shape=None,
        kernel_size=3,
        strides=1,
        padding="same",
        **kwargs,
    ):
        """
        :param filters:
        :param output_shape: [out_dim1, out_dim2, out_dim3]
        :param kernel_size:
        :param strides:
        :param padding:
        :param kwargs:
        """
        super(Deconv3dBlock, self).__init__(**kwargs)
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

    def call(self, inputs, training=None, **kwargs):
        output = self._deconv3d(inputs=inputs)
        output = self._norm(inputs=output, training=training)
        output = self._act(output)
        return output


class Residual3dBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super(Residual3dBlock, self).__init__(**kwargs)
        # init layer variables
        self._conv3d_block = Conv3dBlock(
            filters=filters, kernel_size=kernel_size, strides=strides
        )
        self._conv3d = Conv3d(
            filters=filters, kernel_size=kernel_size, strides=strides, use_bias=False
        )
        self._norm = Norm()
        self._act = Activation()

    def call(self, inputs, training=None, **kwargs):
        return self._act(
            self._norm(
                inputs=self._conv3d(inputs=self._conv3d_block(inputs)),
                training=training,
            )
            + inputs
        )


class DownSampleResnetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, pooling=True, **kwargs):
        super(DownSampleResnetBlock, self).__init__(**kwargs)
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

    def call(self, inputs, training=None, **kwargs):
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
        super(UpSampleResnetBlock, self).__init__(**kwargs)
        # save parameters
        self._filters = filters
        self._concat = concat
        # init layer variables
        self._deconv3d_block = None
        self._conv3d_block = Conv3dBlock(filters=filters, kernel_size=kernel_size)
        self._residual_block = Residual3dBlock(filters=filters, kernel_size=kernel_size)

    def build(self, input_shape):
        super(UpSampleResnetBlock, self).build(input_shape)
        skip_shape = input_shape[1][1:4]
        self._deconv3d_block = Deconv3dBlock(
            filters=self._filters, output_shape=skip_shape, strides=2
        )

    def call(self, inputs, training=None, **kwargs):
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
        output_shape,
        filters,
        kernel_initializer="glorot_uniform",
        activation=None,
        **kwargs,
    ):
        """
        perform a conv and resize
        :param output_shape: [out_dim1, out_dim2, out_dim3]
        :param filters:
        :param kwargs:
        """
        super(Conv3dWithResize, self).__init__(**kwargs)
        # save parameters
        self._output_shape = output_shape
        # init layer variables
        self._conv3d = Conv3d(
            filters=filters,
            strides=1,
            kernel_initializer=kernel_initializer,
            activation=activation,
        )  # if not zero, with init NN, ddf may be too large
        self._resize3d = Resize3d(size=output_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: shape = [batch, dim1, dim2, dim3, channels]
        :param kwargs:
        :return: shape = [batch, out_dim1, out_dim2, out_dim3, channels]
        """
        output = self._conv3d(inputs=inputs)
        if inputs.shape[1:4] != self._output_shape:
            output = self._resize3d(inputs=output)
        return output


class Warping(tf.keras.layers.Layer):
    def __init__(self, fixed_image_size, **kwargs):
        """
        :param fixed_image_size: shape = [f_dim1, f_dim2, f_dim3]
                                 or [f_dim1, f_dim2, f_dim3, ch] with the last channel for features
        :param kwargs:
        """
        super(Warping, self).__init__(**kwargs)
        self._grid_ref = tf.expand_dims(
            layer_util.get_reference_grid(grid_size=fixed_image_size), axis=0
        )  # shape = [1, f_dim1, f_dim2, f_dim3, 3]

    def call(self, inputs, **kwargs):
        """
        wrap an image into a fixed size using ddf
        same functionality as transform of neuron
        https://github.com/adalca/neuron/blob/master/neuron/utils.py
        vol = image
        loc_shift = ddf
        :param inputs: [ddf, image]
                        ddf.shape = [batch, f_dim1, f_dim2, f_dim3, 3]
                        image.shape = [batch, m_dim1, m_dim2, m_dim3]
        :param kwargs:
        :return: shape = [batch, f_dim1, f_dim2, f_dim3]
        """
        grid_warped = self._grid_ref + inputs[0]  # [batch, f_dim1, f_dim2, f_dim3, 3]
        image_warped = layer_util.resample(
            vol=inputs[1], loc=grid_warped
        )  # [batch, f_dim1, f_dim2, f_dim3]
        return image_warped


class IntDVF(tf.keras.layers.Layer):
    def __init__(self, fixed_image_size, num_steps=7, **kwargs):
        """
        :param fixed_image_size: shape = [f_dim1, f_dim2, f_dim3]
        :param num_steps: number of steps for integration
        :param kwargs:
        """
        super(IntDVF, self).__init__(**kwargs)
        self._warping = Warping(fixed_image_size=fixed_image_size)
        self._num_steps = num_steps

    def call(self, inputs, **kwargs):
        """
        given a dvf, calculate ddf
        same as integrate_vec of neuron
        https://github.com/adalca/neuron/blob/master/neuron/utils.py
        :param inputs: dvf, shape = [batch, f_dim1, f_dim2, f_dim3, 3]
        :param kwargs:
        :return: ddf, shape = [batch, f_dim1, f_dim2, f_dim3, 3]
        """
        ddf = inputs / (2 ** self._num_steps)
        for _ in range(self._num_steps):
            ddf += self._warping(inputs=[ddf, ddf])
        return ddf


class Dense(tf.keras.layers.Layer):
    def __init__(self, units, bias_initializer="zeros", **kwargs):
        """
        :param output_shape: [batch, 4, 3]
        :param bias_initializer:
        :param kwargs:
        """
        super(Dense, self).__init__(**kwargs)

        # init layer variables
        self._flatten = tf.keras.layers.Flatten()
        self._dense = tf.keras.layers.Dense(
            units=units, bias_initializer=bias_initializer
        )

    def call(self, inputs, training=None, **kwargs):
        flatten_inputs = self._flatten(inputs)
        return self._dense(flatten_inputs)


"""
local net
"""


class AdditiveUpSampling(tf.keras.layers.Layer):
    def __init__(self, output_shape, stride=2, **kwargs):
        """
        :param output_shape: [out_dim1, out_dim2, out_dim3]
        :param stride:
        :param kwargs:
        """
        super(AdditiveUpSampling, self).__init__(**kwargs)
        # save parameters
        self._stride = stride
        # init layer variables
        self._resize3d = Resize3d(size=output_shape)

    def call(self, inputs, **kwargs):
        """
        :param inputs: shape = [batch, dim1, dim2, dim3, channels]
        :param kwargs:
        :return:
        """
        if inputs.shape[4] % self._stride != 0:
            raise ValueError("The channel dimension can not be divided by the stride")
        output = self._resize3d(inputs=inputs)
        output = tf.split(
            output, num_or_size_splits=self._stride, axis=4
        )  # a list of [batch, out_dim1, out_dim2, out_dim3, channels//stride], num = stride
        output = tf.reduce_sum(
            tf.stack(output, axis=5), axis=5
        )  # [batch, out_dim1, out_dim2, out_dim3, channels//stride]
        return output


class LocalNetResidual3dBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super(LocalNetResidual3dBlock, self).__init__(**kwargs)
        # init layer variables
        self._conv3d = Conv3d(
            filters=filters, kernel_size=kernel_size, strides=strides, use_bias=False
        )
        self._norm = Norm()
        self._act = Activation()

    def call(self, inputs, training=None, **kwargs):
        layer_util.check_inputs(inputs, 2, "ResidualBlock")

        return self._act(
            self._norm(inputs=self._conv3d(inputs=inputs[0]), training=training)
            + inputs[1]
        )


class LocalNetUpSampleResnetBlock(tf.keras.layers.Layer):
    def __init__(self, filters, use_additive_upsampling=True, **kwargs):
        super(LocalNetUpSampleResnetBlock, self).__init__(**kwargs)
        # save parameters
        self._filters = filters
        self._use_additive_upsampling = use_additive_upsampling
        # init layer variables
        self._deconv3d_block = None
        self._additive_upsampling = None
        self._conv3d_block = Conv3dBlock(filters=filters)
        self._residual_block = LocalNetResidual3dBlock(filters=filters, strides=1)

    def build(self, input_shape):
        super(LocalNetUpSampleResnetBlock, self).build(input_shape)
        layer_util.check_inputs(input_shape, 2, "UpSampleResnetBlock build")

        output_shape = input_shape[1][1:4]
        self._deconv3d_block = Deconv3dBlock(
            filters=self._filters, output_shape=output_shape, strides=2
        )
        if self._use_additive_upsampling:
            self._additive_upsampling = AdditiveUpSampling(output_shape=output_shape)

    def call(self, inputs, training=None, **kwargs):
        layer_util.check_inputs(inputs, 2, "UpSampleResnetBlock call")

        inputs_nonskip, inputs_skip = inputs[0], inputs[1]
        h0 = self._deconv3d_block(inputs=inputs_nonskip, training=training)
        if self._use_additive_upsampling:
            h0 += self._additive_upsampling(inputs=inputs_nonskip)
        r1 = h0 + inputs_skip
        r2 = self._conv3d_block(inputs=h0, training=training)
        h1 = self._residual_block(inputs=[r2, r1], training=training)
        return h1
