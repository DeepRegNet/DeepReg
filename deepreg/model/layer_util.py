import itertools

import numpy as np
import tensorflow as tf


def he_normal():
    return tf.keras.initializers.he_normal()


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


def pyramid_combination(x, w_f):
    """
    calculate linear interpolation using values of hypercube corners in dimension n
    :param x: values on the corner, has 2**n tensors of shape [batch, *loc_shape]
    :param w_f: weight of floor points, has n tensors of shape [batch, *loc_shape]
    :return:
    """
    if len(w_f) == 1:
        return x[0] * w_f[0] + x[1] * (1 - w_f[0])
    else:
        return pyramid_combination(x[::2], w_f[:-1]) * w_f[-1] + \
               pyramid_combination(x[1::2], w_f[:-1]) * (1 - w_f[-1])


def resample(vol, loc, interpolation="linear"):
    """
    for each voxel at [b, l1, ..., ln]
    sample_coords[b, l1, ..., ln, :] = [v1, ..., vn], which is the coordinates of a source voxel
    output[b, l1, ..., ln] is therefore the value sampled at [b, v1, ..., vn] in source

    :param vol: shape = [batch, v_dim 1, ..., v_dim n] = [batch, *vol_shape]
    :param loc: shape = [batch, l_dim 1, ..., l_dim m, n] = [batch, *loc_shape, n],
                use `loc` instead of `coords` to make code simpler
    :param interpolation: TODO support nearest
    :return: shape = [batch, s_dim 1, ..., s_dim n]

    difference with neuron's interpn https://github.com/adalca/neuron/blob/master/neuron/utils.py
    1. they dont have batch size
    2. they support more dimensions in source

    TODO allow source to have more dimensions
    """

    # init
    batch_size = vol.shape[0]
    vol_shape = vol.shape[1:]
    loc_shape = loc.shape[1: -1]
    n = loc.shape[-1]  # dimension of vol
    assert n == len(vol_shape)
    if interpolation != "linear":
        raise ValueError("only linear interpolation is supported")

    # clip loc to get anchors and weights
    loc_unstack = tf.unstack(loc, axis=-1)  # n tensors of shape [batch, s_dim 1, ..., s_dim m]
    loc_floor_ceil, weight_floor = [], []
    for d, _loc in enumerate(loc_unstack):
        # using for loop is faster than using list comprehension
        clipped = tf.clip_by_value(_loc, clip_value_min=0, clip_value_max=vol_shape[d] - 1)  # [batch, *loc_shape]
        c_ceil = tf.math.ceil(clipped)
        c_floor = tf.maximum(c_ceil - 1, 0)
        w_floor = c_ceil - clipped

        loc_floor_ceil.append([tf.cast(c_floor, tf.int32),
                               tf.cast(c_ceil, tf.int32)])
        weight_floor.append(w_floor)

    # get vol values on n-dim hypercube corners
    corner_indices = get_n_bits_combinations(n=len(vol_shape))  # 2**n corners, each is of n binary values
    # range(batch_size) on axis 0 and repeated on other axises
    # add batch coords manually is faster than using batch_dims in tf.gather_nd
    batch_coords = tf.tile(tf.reshape(tf.range(batch_size), [batch_size] + [1] * len(loc_shape)),
                           [1] + loc_shape)  # [batch, *loc_shape]
    corner_values = [
        tf.gather_nd(vol,  # shape [batch, *vol_shape]
                     tf.stack([batch_coords] + [loc_floor_ceil[axis][fc_idx]
                                                for axis, fc_idx in enumerate(c)],  # combine batch coord and loc
                              axis=-1))  # get value in vol, shape [batch, *loc_shape, n+1]
        for c in corner_indices  # for each corner
    ]  # each tensor has shape [batch, *loc_shape]

    # resample
    sampled = pyramid_combination(corner_values, weight_floor)
    return sampled


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
