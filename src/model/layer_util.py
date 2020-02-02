import itertools

import numpy as np
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


def conv3d(filters, kernel_size=3, strides=1, padding="same", activation=None, use_bias=True,
           kernel_initializer="glorot_uniform"):
    """
    :param filters: number of channels of the output
    :param kernel_size: e.g. (3,3,3) or 3
    :param strides: e.g. (1,1,1) or 1
    :param padding: "valid" or "same"
    :param activation:
    :param use_bias:
    :param kernel_initializer:
    :return:
    """
    return tf.keras.layers.Conv3D(filters=filters,
                                  kernel_size=kernel_size,
                                  strides=strides,
                                  padding=padding,
                                  activation=activation,
                                  use_bias=use_bias,
                                  kernel_initializer=kernel_initializer, )


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


def get_reference_grid(grid_size):
    """
    :param grid_size: list or tuple of size 3, [dim1, dim2, dim3]
    :return: tf tensor, shape = [dim1, dim2, dim3, 3], grid[i, j, k, :] = [i j k]

    for tf.meshgrid, in the 3-D case with inputs of length M, N and P,
    outputs are of shape (N, M, P) for ‘xy’ indexing and (M, N, P) for ‘ij’ indexing.
    """
    check_inputs(grid_size, 3, "get_reference_grid")
    return tf.cast(tf.stack(tf.meshgrid(
        np.arange(grid_size[0]),
        np.arange(grid_size[1]),
        np.arange(grid_size[2]),
        indexing='ij'), axis=3), dtype=tf.float32)


def get_n_bits_combinations(n):
    return [list(i) for i in itertools.product([0, 1], repeat=n)]


def resample_linear_n(source, sample_coords):
    """

    for each voxel at [b, d1, ..., dn]
    sample_coords[b, d1, ..., dn] = [q1, ..., qn]
    it's the values sampled at [b, q1, ..., qn] in source

    :param source: shape = [batch, g_dim 1, ..., g_dim n]
    :param sample_coords: shape = [batch, s_dim 1, ..., s_dim m, n]
    :param n: source dimension except batch
    :return: shape = [batch, s_dim 1, ..., s_dim n]
    """

    batch_size = source.shape[0]
    grid_dims = source.shape[1:]  # value = [g_dim 1, ..., g_dim n], let their product be g_dims_prod

    coords = tf.unstack(sample_coords, axis=-1)  # n tensors of shape [batch, s_dim 1, ..., s_dim m]
    coords_floor_ceil = []  # n tensors of shape [batch, s_dim 1, ..., s_dim m, 2]
    weight_floor = []  # no need of weight for ceil as their sum is 1
    for axis, c in enumerate(coords):
        clipped = tf.clip_by_value(c, clip_value_min=0, clip_value_max=grid_dims[axis] - 1)  # [batch, s_dims_prod]
        c_ceil = tf.math.ceil(clipped)
        c_floor = tf.maximum(c_ceil - 1, 0)
        w_floor = c_ceil - clipped

        coords_floor_ceil.append([tf.cast(c_floor, tf.int32), tf.cast(c_ceil, tf.int32)])
        weight_floor.append(w_floor)

    corner_indices = get_n_bits_combinations(n=len(grid_dims))  # 2**n corners, each is of n binary values
    corner_values = []
    # add batch coords manually is faster than using batch_dims in tf.gather_nd
    batch_coords = tf.tile(tf.reshape(tf.range(batch_size), [batch_size] + [1] * (len(sample_coords.shape) - 2)),
                           [1] + sample_coords.shape[1:-1])  # [batch, s_dim 1, ..., s_dim m]
    for c_indices in corner_indices:
        c_coords = tf.stack([batch_coords] + [coords_floor_ceil[axis][fc_idx]
                                              for axis, fc_idx in enumerate(c_indices)],
                            axis=-1)  # shape [batch, s_dim 1, ..., s_dim m, n+1]
        c_values = tf.gather_nd(source, c_coords)  # shape [batch, s_dim 1, ..., s_dim m]
        corner_values.append(c_values)

    def pyramid_combination(x, w_f):
        if len(w_f) == 1:
            return x[0] * w_f[0] + x[1] * (1 - w_f[0])
        else:
            return pyramid_combination(x[::2], w_f[:-1]) * w_f[-1] + \
                   pyramid_combination(x[1::2], w_f[:-1]) * (1 - w_f[-1])

    sampled_values = pyramid_combination(corner_values, weight_floor)
    return sampled_values


def resample_linear(inputs, sample_coords):
    """

    :param inputs: shape = [batch, dim1, dim2, dim3]
    :param sample_coords: shape = [batch, dim1, dim2, dim3, 3]
    :return: shape = [batch, dim1, dim2, dim3]
    """

    assert len(inputs.shape) == 4
    res = resample_linear_n(inputs, sample_coords)

    return res


def random_transform_generator(batch_size, scale=0.1):
    """

    :param batch_size:
    :param scale:
    :return: tf tensor, shape = [batch, 4, 3]

    affine transformation

    [[x' y' z' 1]] = [[x y z 1]] * [[* * * 0]
                                    [* * * 0]
                                    [* * * 0]
                                    [* * * 1]]
    new = old * T
    shape: [1, 4] = [1, 4] * [4, 4]

    equivalent to
    [[x' y' z']] = [[x y z 1]] * [[* * *]
                                  [* * *]
                                  [* * *]
                                  [* * *]]
    x, y, z are original coordinates
    x', y', z' are transformed coordinates

    for (x, y, z) the noise is -(x, y, z) .* (r1, r2, r3) where ri is a random number between (0, scale)
    so (x', y', z') = (x, y, z) .* (1-r1, 1-r2, 1-r3)

    this version is faster (0.4ms -> 0.2ms) per call
    """
    noise = np.random.uniform(1 - scale, 1, [batch_size, 4, 3])  # [batch, 4, 3]

    old = np.tile([[[-1, -1, -1, 1],
                    [-1, -1, 1, 1],
                    [-1, 1, -1, 1],
                    [1, -1, -1, 1]]], [batch_size, 1, 1])  # [batch, 4, 4], [0, 0, :] = [-1,-1,-1,1]
    new = old[:, :, :3] * noise  # [batch, 4, 3]

    theta = np.array([np.linalg.lstsq(old[k], new[k], rcond=-1)[0]
                      for k in range(batch_size)])  # [batch, 4, 3]

    return tf.cast(theta, dtype=tf.float32)


def warp_grid(grid, theta):
    """
    perform transformation on the grid

    :param grid: shape = [dim1, dim2, dim3, 3], grid[i,j,k,:] = [i j k]
    :param theta: parameters of transformation, shape = [batch, 4, 3]
    :return: shape = [batch, dim1, dim2, dim3, 3]

    grid_padded[i,j,k,:] = [i j k 1]
    grid_warped[b,i,j,k,p] = sum_over_q (grid_padded[i,j,k,q] * theta[b,q,p])

    using einsum is faster (8ms -> 5ms) per call
    """

    grid_size = grid.get_shape().as_list()
    grid = tf.concat([grid, tf.ones(grid_size[:3] + [1])], axis=3)  # [dim1, dim2, dim3, 4]
    grid_warped = tf.einsum("ijkq,bqp->bijkp", grid, theta)  # [batch, dim1, dim2, dim3, 3]
    return grid_warped
