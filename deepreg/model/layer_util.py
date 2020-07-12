"""
Module containing utilities for layer inputs
"""
import itertools

import numpy as np
import tensorflow as tf


def check_inputs(inputs, size, msg=""):
    """
    Function to assert correct input types and length
    :param inputs: list or tuple
    :param size: int, defines corect size of input tuple/list
    :param msg: string type to return with raised ValueError
    """
    if msg != "":
        msg += " "
    if not isinstance(inputs, (list, tuple)):
        raise ValueError(msg + "Inputs should be a list or tuple")
    if len(inputs) != size:
        raise ValueError(
            msg
            + """Inputs should be a list or tuple\
                     of size %d, but received %d"""
            % (size, len(inputs))
        )


def get_reference_grid(grid_size: (tuple, list)) -> tf.Tensor:
    """
    :param grid_size: list or tuple of size 3, [dim1, dim2, dim3]
    :return: shape = [dim1, dim2, dim3, 3],
             grid[i, j, k, :] = [i j k]

    for tf.meshgrid, in the 3-D case with inputs of length M, N and P,
    outputs are of shape (N, M, P) for ‘xy’ indexing and
    (M, N, P) for ‘ij’ indexing.

    same function as volshape_to_meshgrid of neuron
    https://github.com/adalca/neuron/blob/master/neuron/utils.py
    neuron modifies meshgrid to make it faster, however local
    benchmark suggests tf.meshgrid is better
    """

    # dim1, dim2, dim3 = grid_size
    # mesh_grid has three elements, corresponding to i, j, k
    # for i in range(dim1)
    #     for j in range(dim2)
    #         for k in range(dim3)
    #             mesh_grid[0][i,j,k] = i
    #             mesh_grid[1][i,j,k] = j
    #             mesh_grid[2][i,j,k] = k
    mesh_grid = tf.meshgrid(
        tf.range(grid_size[0]),
        tf.range(grid_size[1]),
        tf.range(grid_size[2]),
        indexing="ij",
    )  # has three elements, each shape = (dim1, dim2, dim3)
    grid = tf.stack(mesh_grid, axis=3)  # shape = (dim1, dim2, dim3, 3)
    grid = tf.cast(grid, dtype=tf.float32)
    return grid


def get_n_bits_combinations(num_bits: int) -> list:
    """
    Function returning list containing all combinations of n bits.
    Given num_bits binary bits, each bit has value 0 or 1,
    there are in total 2**n_bits combinations
    :param num_bits: int, number of combinations to evaluate
    :return: a list of length 2**n_bits
    return[i] is the binary representation of the decimal integer
    for example, when num_bits = 3
    get_n_bits_combinations(3) =
    [
        [0, 0, 0], # 0
        [0, 0, 1], # 1
        [0, 1, 0], # 2
        [0, 1, 1], # 3
        [1, 0, 0], # 4
        [1, 0, 1], # 5
        [1, 1, 0], # 6
        [1, 1, 1], # 7
    ]
    """
    assert num_bits >= 1
    return [list(i) for i in itertools.product([0, 1], repeat=num_bits)]


def pyramid_combination(values: list, weights: list) -> list:
    """
    Calculate linear interpolation (a weighted sum) using values of
    hypercube corners in dimension n.

    :param values: a list having values on the corner,
                   it has 2**n tensors of shape
                   (*loc_shape) or (batch, *loc_shape) or (batch, *loc_shape, ch)
                   the order is consistent with get_n_bits_combinations
                   loc_shape is independent from n, aka num_dim
    :param weights: a list having weights of floor points,
                    it has n tensors of shape
                    (*loc_shape) or (batch, *loc_shape) or (batch, *loc_shape, 1)
    :return: one tensor of the same shape as an element in values
             (*loc_shape) or (batch, *loc_shape) or (batch, *loc_shape, 1)

    for example, when num_dimension = len(loc_shape) = num_bits = 3
    values correspond to values at corners of following coordinates
    [
        [0, 0, 0], # even
        [0, 0, 1], # odd
        [0, 1, 0], # even
        [0, 1, 1], # odd
        [1, 0, 0], # even
        [1, 0, 1], # odd
        [1, 1, 0], # even
        [1, 1, 1], # odd
    ]
    values[::2] correspond to the corners with last coordinate == 0
    [
        [0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0],
    ]
    values[1::2] correspond to the corners with last coordinate == 1
    [
        [0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]

    the weights correspond to the floor corners,
    for example, when num_dimension = len(loc_shape) = num_bits = 3
    weights = [w1, w2, w3] (ignoring the batch dimension)
    so for corner with coords (x, y, z), x, y, z's values are 0 or 1
        weight for x = w1 if x = 0 else 1-w1
        weight for y = w2 if y = 0 else 1-w2
        weight for z = w3 if z = 0 else 1-w3
    so the weight for (x, y, z), denoted by W_xyz is
          ((1-x) * w1 + x * (1-w1))
        * ((1-y) * w2 + y * (1-w2))
        * ((1-z) * w3 + z * (1-w3))
    which equals
        the weight for (x, y) * ((1-z) * w3 + z * (1-w3))
    let W_xy = the weight for (x, y)
    W_xyz equals
          (W_xy * (1-z)) * w3
        + (W_xy * z) * (1-w3)
    So W_xy0 = W_xy * w3
       W_xy1 = W_xy * (1-w3)

    So V = sum over x,y,z (V_xyz * W_xyz)
         = sum over x,y ( V_xy0 * W_xy0 + V_xy1 * W_xy1 )
         = sum over x,y ( V_xy0 * W_xy * w3 + V_xy1 * W_xy * (1-w3) )
         = sum over x,y ( W_xy * (V_xy0 * w3 + V_xy1 * W_xy * (1-w3)) )

    That's why we call this pyramid combination,
    it calculates the linear interpolation gradually, starting from
    the last dimension.
    The key is that the weight of each corner is the product of the weights
    along each dimension.
    """
    if len(values[0].shape) != len(weights[0].shape):
        raise ValueError(
            """In pyramid_combination, elements of values and
            weights should have same dimension.
            value shape = {}, weight = {}""".format(
                values[0].shape, weights[0].shape
            )
        )
    if 2 ** len(weights) != len(values):
        raise ValueError(
            "In pyramid_combination, "
            "num_dim = len(weights), "
            "len(values) must be 2 ** num_dim, "
            "But len(weights) = {}, len(values) = {}".format(len(weights), len(values))
        )

    if len(weights) == 1:  # one dimension
        return values[0] * weights[0] + values[1] * (1 - weights[0])
    # multi dimension
    values_floor = pyramid_combination(values[::2], weights[:-1]) * weights[-1]
    values_ceil = pyramid_combination(values[1::2], weights[:-1]) * (1 - weights[-1])
    return values_floor + values_ceil


def resample(vol, loc, interpolation="linear"):
    """
    Given a volume, vol, of shape (batch, v_dim 1, ..., v_dim n),
    where n is the dimension of volume,
    we want to sample multiple locations, where the coordinates are provided in
    loc, which has shape (batch, l_dim 1, ..., l_dim m, n),
    where m is the dimension of output

    loc[b, l1, ..., ln, :] = [v1, ..., vn] is of shape (n,)
    it represents a point in vol, with coordinates
    (v1, ..., vn)

    the output is of shape (batch, l_dim 1, ..., l_dim n),
    corresponds the samples inside the given volume

    The volume can also have an extra dimension as features.

    :param vol: shape = (batch, v_dim 1, ..., v_dim n)
                      = (batch, *vol_shape)
                or shape = (batch, v_dim 1, ..., v_dim n, ch)
                         = (batch, *vol_shape, ch)
                with the last channel for features
    :param loc: shape = [batch, l_dim 1, ..., l_dim m, n] =
                        [batch, *loc_shape, n],
    :param interpolation: linear only, TODO support nearest
    :return: shape = (batch, l_dim 1, ..., l_dim n)

    difference with neuron's interpn
    https://github.com/adalca/neuron/blob/master/neuron/utils.py
    1. they dont have batch size
    2. they support more dimensions in vol

    TODO try not using stack as neuron claims it's slower
    """

    if interpolation != "linear":
        raise ValueError("resample supports only linear interpolation")

    # init
    batch_size = vol.shape[0]
    loc_shape = loc.shape[1:-1]
    dim_vol = loc.shape[-1]  # dimension of vol
    if dim_vol == len(vol.shape) - 1:
        # vol.shape = (batch, *vol_shape)
        has_ch = False
    elif dim_vol == len(vol.shape) - 2:
        # vol.shape = (batch, *vol_shape, ch)
        has_ch = True
    else:
        raise ValueError(
            "vol shape inconsistent with loc "
            "vol.shape = {}, loc.shape = {}".format(vol.shape, loc.shape)
        )
    vol_shape = vol.shape[1 : dim_vol + 1]

    # clip loc to get anchors and weights
    # loc_unstack has n tensors of shape (batch, l_dim 1, ..., l_dim m)
    # the d-th tensor corresponds to the coordinates of d-th dimension

    # loc_floor_ceil has n sublists
    # each one corresponds to the floor and ceil coordinates for d-th dimension
    # each tensor is of shape (batch, *loc_shape), dtype int32

    # weight_floor has n tensors
    # each tensor is the weight for the corner of floor coordinates
    # each tensor's shape is (batch, *loc_shape) if volume has no feature channel
    #                        (batch, *loc_shape, 1) if volume has feature channel
    loc_unstack = tf.unstack(loc, axis=-1)
    loc_floor_ceil, weight_floor = [], []
    for dim, loc_d in enumerate(loc_unstack):
        # using for loop is faster than using list comprehension
        # clip to be inside 0 ~ (l_dim d - 1)
        clipped = tf.clip_by_value(
            loc_d, clip_value_min=0, clip_value_max=vol_shape[dim] - 1
        )  # shape = (batch, *loc_shape)
        c_ceil = tf.math.ceil(clipped)  # shape = (batch, *loc_shape)
        c_floor = tf.maximum(c_ceil - 1, 0)  # shape = (batch, *loc_shape)
        w_floor = c_ceil - clipped  # shape = (batch, *loc_shape)
        if has_ch:
            w_floor = tf.expand_dims(w_floor, -1)  # shape = (batch, *loc_shape, 1)
        loc_floor_ceil.append([tf.cast(c_floor, tf.int32), tf.cast(c_ceil, tf.int32)])
        weight_floor.append(w_floor)

    # 2**n corners, each is a list of n binary values
    corner_indices = get_n_bits_combinations(num_bits=len(vol_shape))

    # batch_coords[b, l1, ..., lm] = b
    # range(batch_size) on axis 0 and repeated on other axises
    # add batch coords manually is faster than using batch_dims in tf.gather_nd
    batch_coords = tf.tile(
        tf.reshape(tf.range(batch_size), [batch_size] + [1] * len(loc_shape)),
        [1] + loc_shape,
    )  # shape = (batch, *loc_shape)

    # get vol values on n-dim hypercube corners
    # corner_values has 2 ** n elements
    # each of shape (batch, *loc_shape) or (batch, *loc_shape, ch)
    corner_values = [
        tf.gather_nd(
            vol,  # shape = (batch, *vol_shape) or (batch, *vol_shape, ch)
            tf.stack(
                [batch_coords]
                + [loc_floor_ceil[axis][fc_idx] for axis, fc_idx in enumerate(c)],
                axis=-1,
            ),  # shape = (batch, *loc_shape, n+1) after stack
        )
        for c in corner_indices  # c is list of len n
    ]  # each tensor has shape (batch, *loc_shape) or (batch, *loc_shape, ch)

    # resample
    sampled = pyramid_combination(corner_values, weight_floor)
    return sampled


def random_transform_generator(
    batch_size: int, scale: float, seed: (int, None) = None
) -> tf.Tensor:
    """
    Function that generates a random 3D transformation parameters for a batch of data

    :param batch_size: int
    :param scale: a float number between 0 and 1
    :param seed: control the randomness
    :return: shape = (batch, 4, 3)

    for 3D coordinates, affine transformation is
    [[x' y' z' 1]] = [[x y z 1]] * [[* * * 0]
                                    [* * * 0]
                                    [* * * 0]
                                    [* * * 1]]
    where each * represents a degree of freedom,
    so there are in total 12 degrees of freedom
    the equation can be denoted as
        new = old * T
    where
    - new is the transformed coordinates, of shape (1, 4)
    - old is the original coordinates, of shape (1, 4)
    - T is the transformation matrix, of shape (4, 4)

    the equation can be simplified to
    [[x' y' z']] = [[x y z 1]] * [[* * *]
                                  [* * *]
                                  [* * *]
                                  [* * *]]
    so that
        new = old * T
    where
    - new is the transformed coordinates, of shape (1, 3)
    - old is the original coordinates, of shape (1, 4)
    - T is the transformation matrix, of shape (4, 3)

    Given original and transformed coordinates,
    we can calculate the transformation matrix using
        x = np.linalg.lstsq(a, b)
    such that
        a x = b
    in our case,
    - a = old
    - b = new
    - x = T

    To generate random transformation,
    we choose to add random perturbation to corner coordinates as follows:
    for corner of coordinates (x, y, z), the noise is
        -(x, y, z) .* (r1, r2, r3)
    where ri is a random number between (0, scale)
    so
        (x', y', z') = (x, y, z) .* (1-r1, 1-r2, 1-r3)
    thus, we can directly sample between 1-scale and 1 instead

    we choose to calculate the transformation based on
    four corners in a cube centered at (0, 0, 0)
    a cube is shown as below
    where
    - C = (-1, -1, -1)
    - G = (-1, -1, 1)
    - D = (-1, 1, -1)
    - A = (1, -1, -1)

                G — — — — — — — — H
              / |               / |
            /   |             /   |
          /     |           /     |
        /       |         /       |
      /         |       /         |
    E — — — — — — — — F           |
    |           |     |           |
    |           |     |           |
    |           C — — | — — — — — D
    |         /       |         /
    |       /         |       /
    |     /           |     /
    |   /             |   /
    | /               | /
    A — — — — — — — — B
    """

    assert 0 <= scale <= 1
    np.random.seed(seed=seed)
    noise = np.random.uniform(1 - scale, 1, [batch_size, 4, 3])  # shape = (batch, 4, 3)

    # old represents four corners of a cube
    # corresponding to the corner C G D A as shown above
    old = np.tile(
        [[[-1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]],
        [batch_size, 1, 1],
    )  # shape = (batch, 4, 4)
    new = old[:, :, :3] * noise  # shape = (batch, 4, 3)

    theta = np.array(
        [np.linalg.lstsq(old[k], new[k], rcond=-1)[0] for k in range(batch_size)]
    )  # shape = (batch, 4, 3)

    return tf.cast(theta, dtype=tf.float32)


def warp_grid(grid: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    """
    perform transformation on the grid

    :param grid: shape = (dim1, dim2, dim3, 3), grid[i,j,k,:] = [i j k]
    :param theta: parameters of transformation, shape = (batch, 4, 3)
    :return: shape = (batch, dim1, dim2, dim3, 3)

    grid_padded[i,j,k,:] = [i j k 1]
    grid_warped[b,i,j,k,p] = sum_over_q (grid_padded[i,j,k,q] * theta[b,q,p])
    """

    grid_size = grid.get_shape().as_list()

    # grid_padded[i,j,k,:] = [i j k 1], shape = (dim1, dim2, dim3, 4)
    grid_padded = tf.concat([grid, tf.ones(grid_size[:3] + [1])], axis=3)

    # grid_warped[b,i,j,k,p] = sum_over_q (grid_padded[i,j,k,q] * theta[b,q,p])
    # shape = (batch, dim1, dim2, dim3, 3)
    grid_warped = tf.einsum("ijkq,bqp->bijkp", grid_padded, theta)
    return grid_warped


def resize3d(
    image: tf.Tensor, size: (tuple, list), method: str = tf.image.ResizeMethod.BILINEAR
) -> tf.Tensor:
    """
    tensorflow does not have resize 3d, therefore the resize is performed two folds.
    - resize dim2 and dim3
    - resize dim1 and dim2
    :param image: tensor of shape = (batch, dim1, dim2, dim3, channels) or (batch, dim1, dim2, dim3)
    :param size: tuple, (out_dim1, out_dim2, out_dim3)
    :param method: str, one of tf.image.ResizeMethod
    :return: tensor of shape = (batch, out_dim1, out_dim2, out_dim3, channels) or shape = (batch, dim1, dim2, dim3)
    """

    if len(image.shape) not in [4, 5]:
        raise ValueError(
            "resize3d takes input image of dimension 4 or 5,"
            "corresponding to (batch, dim1, dim2, dim3, channels) or (batch, dim1, dim2, dim3),"
            "got image shape{}".format(image.shape)
        )
    if len(size) != 3:
        raise ValueError("resize3d takes size of type tuple/list and of length 3")
    has_channel = len(image.shape) == 5
    if not has_channel:  # convert to channel mode
        image = tf.expand_dims(image, axis=4)
    image_shape = image.shape

    # merge axis 0 and 1
    output = tf.reshape(
        image, [-1, image_shape[2], image_shape[3], image_shape[4]]
    )  # [batch * dim1, dim2, dim3, channels]
    # resize dim2 and dim3
    output = tf.image.resize(
        images=output, size=[size[1], size[2]], method=method
    )  # [batch * dim1, out_dim2, out_dim3, channels]

    # split axis 0 and merge axis 3 and 4
    output = tf.reshape(
        output, shape=[-1, image_shape[1], size[1], size[2] * image_shape[4]]
    )  # [batch, dim1, out_dim2, out_dim3 * channels]
    # resize dim1 and dim2
    output = tf.image.resize(
        images=output, size=[size[0], size[1]], method=method
    )  # [batch, out_dim1, out_dim2, out_dim3 * channels]
    # reshape
    output = tf.reshape(
        output, shape=[-1, size[0], size[1], size[2], image_shape[4]]
    )  # [batch, out_dim1, out_dim2, out_dim3, channels]

    if not has_channel:  # remove the channel axis
        output = tf.squeeze(output, axis=4)
    return output
