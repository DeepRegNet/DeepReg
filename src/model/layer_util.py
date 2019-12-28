import tensorflow as tf


def he_normal():
    return tf.keras.initializers.he_normal()


def act(identifier):
    """
    :param identifier: e.g. "relu"
    :return:
    """
    return tf.keras.activations.get(identifier=identifier)


def batch_norm(axis=-1):
    """
    :param axis: the axis that should be normalized (typically the features axis)
    :return:
    """
    return tf.keras.layers.BatchNormalization(axis=axis)


def conv3d(filters, kernel_size=3, strides=1, padding="same", activation=None, use_bias=True):
    """
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
                                  kernel_initializer=he_normal())


def max_pool3d(pool_size, strides=None, padding="same"):
    """
    :param pool_size: e.g. (2,2,2)
    :param strides: e.g. (2.2.2)
    :param padding:
    :return:
    """
    return tf.keras.layers.MaxPool3D(pool_size=pool_size,
                                     strides=strides,
                                     padding=padding)


def check_inputs(inputs, size, msg=""):
    if msg != "":
        msg += " "
    if not (isinstance(inputs, list) or isinstance(inputs, tuple)):
        raise ValueError(msg + "Inputs should be a list or tuple")
    if len(inputs) != size:
        raise ValueError(msg + "Inputs should be a list or tuple of size %d, but received %d" % (size, len(inputs)))


def warp_moving(moving_image_or_label, grid_warped):
    return resample_linear(moving_image_or_label, grid_warped)


def get_reference_grid(grid_size):
    return tf.cast(tf.stack(tf.meshgrid(
        [i for i in range(grid_size[0])],
        [j for j in range(grid_size[1])],
        [k for k in range(grid_size[2])],
        indexing='ij'), axis=3), dtype=tf.float32)


def resample_linear(inputs, sample_coords):
    input_size = inputs.get_shape().as_list()[1:-1]
    spatial_rank = inputs.get_shape().ndims - 2
    xy = tf.unstack(sample_coords, axis=len(sample_coords.get_shape()) - 1)
    index_voxel_coords = [tf.floor(x) for x in xy]

    def boundary_replicate(sample_coords0, input_size0):
        return tf.maximum(tf.minimum(sample_coords0, input_size0 - 1), 0)

    spatial_coords = [boundary_replicate(tf.cast(x, tf.int32), input_size[idx])
                      for idx, x in enumerate(index_voxel_coords)]
    spatial_coords_plus1 = [boundary_replicate(tf.cast(x + 1., tf.int32), input_size[idx])
                            for idx, x in enumerate(index_voxel_coords)]

    weight = [tf.expand_dims(x - tf.cast(i, tf.float32), -1) for x, i in zip(xy, spatial_coords)]
    weight_c = [tf.expand_dims(tf.cast(i, tf.float32) - x, -1) for x, i in zip(xy, spatial_coords_plus1)]

    sz = spatial_coords[0].get_shape().as_list()
    batch_coords = tf.tile(tf.reshape(tf.range(sz[0]), [sz[0]] + [1] * (len(sz) - 1)), [1] + sz[1:])
    sc = (spatial_coords, spatial_coords_plus1)
    binary_codes = [[int(c) for c in format(i, '0%ib' % spatial_rank)] for i in range(2 ** spatial_rank)]

    make_sample = lambda bc: tf.gather_nd(inputs, tf.stack([batch_coords] + [sc[c][i] for i, c in enumerate(bc)], -1))
    samples = [make_sample(bc) for bc in binary_codes]

    def pyramid_combination(samples0, weight0, weight_c0):
        if len(weight0) == 1:
            return samples0[0] * weight_c0[0] + samples0[1] * weight0[0]
        else:
            return pyramid_combination(samples0[::2], weight0[:-1], weight_c0[:-1]) * weight_c0[-1] + \
                   pyramid_combination(samples0[1::2], weight0[:-1], weight_c0[:-1]) * weight0[-1]

    return pyramid_combination(samples, weight, weight_c)
