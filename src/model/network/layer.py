import tensorflow as tf


def he_normal():
    return tf.keras.initializers.he_normal()


def act(inputs, identifier):
    """
    :param inputs:
    :param identifier: e.g. "relu"
    :return:
    """
    return tf.keras.activations.get(identifier=identifier)(inputs)


def batch_norm(inputs, training, axis=-1):
    """
    :param inputs:
    :param training:
    :param axis: the axis that should be normalized (typically the features axis)
    :return:
    """
    return tf.keras.layers.BatchNormalization(axis=axis)(inputs, training=training)


def conv3d(inputs, filters, kernel_size=3, strides=1, padding="same", activation=None, use_bias=True):
    """
    :param inputs: shape = [batch, conv_dim1, conv_dim2, conv_dim3, channels]
    :param filters: number of channels of the output
    :param kernel_size: e.g. (3,3,3) or 3
    :param strides: e.g. (1,1,1) or 1
    :param padding: "valid" or "same"
    :param activation:
    :param use_bias:
    :return:
    """
    return tf.keras.layers.Conv3D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  activation=activation,
                                  use_bias=use_bias,
                                  kernel_initializer=he_normal())(inputs=inputs)


def deconv3d(inputs, filters, kernel_size=3, strides=1, padding="same", activation=None, use_bias=True):
    """
    :param inputs: shape = [batch, conv_dim1, conv_dim2, conv_dim3, channels]
    :param filters: number of channels of the output
    :param kernel_size: e.g. (3,3,3) or 3
    :param strides: e.g. (1,1,1) or 1
    :param padding: "valid" or "same"
    :param activation:
    :param use_bias:
    :return:
    """
    return tf.keras.layers.Conv3DTranspose(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=strides,
                                           padding=padding,
                                           activation=activation,
                                           use_bias=use_bias,
                                           kernel_initializer=he_normal())(inputs=inputs)


# def deconv3d_with_output_shape(inputs, filters, output_shape=None, kernel_size=3, strides=1, padding="same"):
#     """
#     :param inputs: shape = [batch, conv_dim1, conv_dim2, conv_dim3, channels]
#     :param filters: number of channels of the output
#     :param output_shape: specify the output shape for axis 1, 2, 3
#     :param kernel_size: e.g. (3,3,3) or 3
#     :param strides: e.g. (1,1,1) or 1
#     :param padding: "valid" or "same"
#     :return:
#     """
#     tf.Variable()
#     if isinstance(kernel_size, int):
#         kernel_size = (kernel_size, kernel_size, kernel_size)
#     if isinstance(strides, int):
#         strides = (strides, strides, strides)
#     output_padding = None
#     if output_shape is not None:
#         padding = "valid"
#         output_padding = [output_shape[i] - strides[i] * (inputs.shape[1 + i] - 1) for i in range(3)]
#         # sanity check
#         for i in range(3):
#             if output_padding[i] < 0 or output_padding[i] > strides[i]:
#                 raise ValueError(
#                     "Deconv3d output shape invalid.",
#                     inputs.shape, output_shape, kernel_size, strides, padding, output_padding)
#     return tf.keras.layers.Conv3DTranspose(filters=filters,
#                                            kernel_size=kernel_size,
#                                            strides=strides,
#                                            padding=padding,
#                                            output_padding=output_padding,
#                                            activation=activation,
#                                            use_bias=use_bias,
#                                            kernel_initializer=he_normal())(inputs=inputs)


# def deconv3d(inputs, filters,
#              output_shape=None, kernel_size=3, strides=1, padding="same", activation=None, use_bias=True):
#     """
#     :param inputs: shape = [batch, conv_dim1, conv_dim2, conv_dim3, channels]
#     :param filters: number of channels of the output
#     :param output_shape: specify the output shape for axis 1, 2, 3
#     :param kernel_size: e.g. (3,3,3) or 3
#     :param strides: e.g. (1,1,1) or 1
#     :param padding: "valid" or "same"
#     :param activation:
#     :param use_bias:
#     :return:
#     """
#     if isinstance(kernel_size, int):
#         kernel_size = (kernel_size, kernel_size, kernel_size)
#     if isinstance(strides, int):
#         strides = (strides, strides, strides)
#     output_padding = None
#     if output_shape is not None:
#         padding = "valid"
#         output_padding = [output_shape[i] - strides[i] * (inputs.shape[1 + i] - 1) for i in range(3)]
#         # sanity check
#         for i in range(3):
#             if output_padding[i] < 0 or output_padding[i] > strides[i]:
#                 raise ValueError(
#                     "Deconv3d output shape invalid.",
#                     inputs.shape, output_shape, kernel_size, strides, padding, output_padding)
#     return tf.keras.layers.Conv3DTranspose(filters=filters,
#                                            kernel_size=kernel_size,
#                                            strides=strides,
#                                            padding=padding,
#                                            output_padding=output_padding,
#                                            activation=activation,
#                                            use_bias=use_bias,
#                                            kernel_initializer=he_normal())(inputs=inputs)


def max_pool3d(inputs, pool_size, strides=None, padding="same"):
    """
    :param inputs:
    :param pool_size: e.g. (2,2,2)
    :param strides: e.g. (2.2.2)
    :param padding:
    :return:
    """
    return tf.keras.layers.MaxPool3D(pool_size=pool_size,
                                     strides=strides,
                                     padding=padding)(inputs=inputs)


def resize3d(inputs, size, method=tf.image.ResizeMethod.BILINEAR):
    """
    tensorflow does not have resize 3d, therefore the resize is performed two folds.
    - resize dim2 and dim3
    - resize dim1 and dim2
    :param inputs: shape = [batch, dim1, dim2, dim3, channels]
    :param size: [out_dim1, out_dim2, out_dim3], list or tuple
    :param method:
    :return:
    """

    in_shape = inputs.shape

    # merge axis 0 and 1
    output = tf.reshape(inputs, [in_shape[0] * in_shape[1],
                                 in_shape[2], in_shape[3], in_shape[4]])  # [batch * dim1, dim2, dim3, channels]
    # resize dim2 and dim3
    output = tf.image.resize(images=output,
                             size=[size[1], size[2]],
                             method=method)  # [batch * dim1, out_dim2, out_dim3, channels]

    # split axis 0 and merge axis 3 and 4
    output = tf.reshape(output,
                        [in_shape[0], in_shape[1], size[1],
                         size[2] * in_shape[4]])  # [batch, dim1, out_dim2, out_dim3 * channels]
    # resize dim1 and dim2
    output = tf.image.resize(images=output,
                             size=[size[0], size[1]],
                             method=method)  # [batch, out_dim1, out_dim2, out_dim3 * channels]
    # reshape
    output = tf.reshape(output, [in_shape[0], size[0], size[1], size[2], in_shape[4]])
    return output  # [batch, out_dim1, out_dim2, out_dim3, channels]


def additive_up_sampling(inputs, size, stride=2):
    """
    :param inputs: shape = [batch, dim1, dim2, dim3, channels]
    :param size: [out_dim1, out_dim2, out_dim3], list or tuple
    :param stride:
    :return:
    """
    if inputs.shape[4] % stride != 0:
        raise ValueError("The channel dimension can not be divided by the stride")
    output = resize3d(inputs=inputs, size=size)  # [batch, out_dim1, out_dim2, out_dim3, channels]
    print("add", output.shape)
    # TODO why not reshape?
    output = tf.split(output,
                      num_or_size_splits=stride,
                      axis=4)  # a list of [batch, out_dim1, out_dim2, out_dim3, channels//stride], num = stride
    output = tf.reduce_sum(tf.stack(output, axis=5), axis=5)  # [batch, out_dim1, out_dim2, out_dim3, channels//stride]
    print("add", output.shape)
    return output


def conv3d_block(inputs, training, filters, kernel_size=3, strides=1, padding="same"):
    output = conv3d(inputs=inputs,
                    filters=filters,
                    kernel_size=kernel_size,
                    strides=strides,
                    padding=padding)
    output = batch_norm(inputs=output, training=training)
    output = act(inputs=output, identifier="relu")
    return output


def deconv3d_block(inputs, training, filters, output_shape=None, kernel_size=3, strides=1, padding="same"):
    output = deconv3d(inputs=inputs,
                      filters=filters,
                      # output_shape=output_shape,
                      kernel_size=kernel_size,
                      strides=strides,
                      padding=padding)
    output = batch_norm(inputs=output, training=training)
    output = act(inputs=output, identifier="relu")
    return output


def downsample_resnet_block(inputs, training, filters, kernel_size=3, use_pooling=True):
    h0 = conv3d_block(inputs=inputs, training=training, filters=filters, kernel_size=kernel_size)
    r1 = conv3d_block(inputs=h0, training=training, filters=filters, kernel_size=kernel_size)
    r2 = act(
        inputs=batch_norm(inputs=conv3d(inputs=r1, filters=filters, kernel_size=kernel_size, strides=1),
                          training=training) + h0,
        identifier="relu")
    if use_pooling:
        h1 = max_pool3d(inputs=r2, pool_size=(2, 2, 2), strides=(2, 2, 2))
    else:
        h1 = conv3d_block(inputs=r2, training=training, filters=filters, kernel_size=kernel_size, strides=2)
    return h1, h0


def upsample_resnet_block(inputs, inputs_skip, training, filters, use_additive_upsampling=True):
    output_shape = inputs_skip.shape[1:4]
    h0 = deconv3d_block(inputs=inputs, training=training, filters=filters, output_shape=output_shape, strides=2)
    if use_additive_upsampling:
        print(inputs.shape, inputs_skip.shape)
        h0 += additive_up_sampling(inputs=inputs, size=output_shape)
    r1 = h0 + inputs_skip
    r2 = conv3d_block(inputs=h0, training=training, filters=filters)
    h1 = act(
        inputs=batch_norm(inputs=conv3d(inputs=r2, filters=filters, strides=1),
                          training=training) + r1,
        identifier="relu")
    return h1


def ddf_summand(inputs, shape_out, filters=3):
    output = conv3d(inputs=inputs, filters=filters, strides=1)
    if inputs.shape != shape_out:  # TODO data type?
        output = resize3d(output, size=shape_out)
    return output
