"""
Module containing utilities for layer inputs
"""
import itertools

import numpy as np
import tensorflow as tf


def get_reference_grid(grid_size: (tuple, list)) -> tf.Tensor:
    """
    Generate a 3D grid with given size.

    Reference:

    - volshape_to_meshgrid of neuron
      https://github.com/adalca/neurite/blob/legacy/neuron/utils.py

      neuron modifies meshgrid to make it faster, however local
      benchmark suggests tf.meshgrid is better

    Note:

    for tf.meshgrid, in the 3-D case with inputs of length M, N and P,
    outputs are of shape (N, M, P) for ‘xy’ indexing and
    (M, N, P) for ‘ij’ indexing.

    :param grid_size: list or tuple of size 3, [dim1, dim2, dim3]
    :return: shape = (dim1, dim2, dim3, 3),
             grid[i, j, k, :] = [i j k]
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
    there are in total 2**n_bits combinations.

    :param num_bits: int, number of combinations to evaluate
    :return: a list of length 2**n_bits,
      return[i] is the binary representation of the decimal integer.

    :Example:
        >>> from deepreg.model.layer_util import get_n_bits_combinations
        >>> get_n_bits_combinations(3)
        [[0, 0, 0], # 0
         [0, 0, 1], # 1
         [0, 1, 0], # 2
         [0, 1, 1], # 3
         [1, 0, 0], # 4
         [1, 0, 1], # 5
         [1, 1, 0], # 6
         [1, 1, 1]] # 7
    """
    assert num_bits >= 1
    return [list(i) for i in itertools.product([0, 1], repeat=num_bits)]


def pyramid_combination(
    values: list, weight_floor: list, weight_ceil: list
) -> tf.Tensor:
    r"""
    Calculates linear interpolation (a weighted sum) using values of
    hypercube corners in dimension n.

    For example, when num_dimension = len(loc_shape) = num_bits = 3
    values correspond to values at corners of following coordinates

    .. code-block:: python

        [[0, 0, 0], # even
         [0, 0, 1], # odd
         [0, 1, 0], # even
         [0, 1, 1], # odd
         [1, 0, 0], # even
         [1, 0, 1], # odd
         [1, 1, 0], # even
         [1, 1, 1]] # odd

    values[::2] correspond to the corners with last coordinate == 0

    .. code-block:: python

        [[0, 0, 0],
         [0, 1, 0],
         [1, 0, 0],
         [1, 1, 0]]

    values[1::2] correspond to the corners with last coordinate == 1

    .. code-block:: python

        [[0, 0, 1],
         [0, 1, 1],
         [1, 0, 1],
         [1, 1, 1]]

    The weights correspond to the floor corners.
    For example, when num_dimension = len(loc_shape) = num_bits = 3,
    weight_floor = [f1, f2, f3] (ignoring the batch dimension).
    weight_ceil = [c1, c2, c3] (ignoring the batch dimension).

    So for corner with coords (x, y, z), x, y, z's values are 0 or 1

    - weight for x = f1 if x = 0 else c1
    - weight for y = f2 if y = 0 else c2
    - weight for z = f3 if z = 0 else c3

    so the weight for (x, y, z) is

    .. code-block:: text

        W_xyz = ((1-x) * f1 + x * c1)
              * ((1-y) * f2 + y * c2)
              * ((1-z) * f3 + z * c3)

    Let

    .. code-block:: text

        W_xy = ((1-x) * f1 + x * c1)
             * ((1-y) * f2 + y * c2)

    Then

    - W_xy0 = W_xy * f3
    - W_xy1 = W_xy * c3

    Similar to W_xyz, denote V_xyz the value at (x, y, z),
    the final sum V equals

    .. code-block:: text

          sum over x,y,z (V_xyz * W_xyz)
        = sum over x,y (V_xy0 * W_xy0 + V_xy1 * W_xy1)
        = sum over x,y (V_xy0 * W_xy * f3 + V_xy1 * W_xy * c3)
        = sum over x,y (V_xy0 * W_xy) * f3 + sum over x,y (V_xy1 * W_xy) * c3

    That's why we call this pyramid combination.
    It calculates the linear interpolation gradually, starting from
    the last dimension.
    The key is that the weight of each corner is the product of the weights
    along each dimension.

    :param values: a list having values on the corner,
                   it has 2**n tensors of shape
                   (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, ch)
                   the order is consistent with get_n_bits_combinations
                   loc_shape is independent from n, aka num_dim
    :param weight_floor: a list having weights of floor points,
                    it has n tensors of shape
                    (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, 1)
    :param weight_ceil: a list having weights of ceil points,
                    it has n tensors of shape
                    (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, 1)
    :return: one tensor of the same shape as an element in values
             (\*loc_shape) or (batch, \*loc_shape) or (batch, \*loc_shape, 1)
    """
    v_shape = values[0].shape
    wf_shape = weight_floor[0].shape
    wc_shape = weight_ceil[0].shape
    if len(v_shape) != len(wf_shape) or len(v_shape) != len(wc_shape):
        raise ValueError(
            "In pyramid_combination, elements of "
            "values, weight_floor, and weight_ceil should have same dimension. "
            f"value shape = {v_shape}, "
            f"weight_floor = {wf_shape}, "
            f"weight_ceil = {wc_shape}."
        )
    if 2 ** len(weight_floor) != len(values):
        raise ValueError(
            "In pyramid_combination, "
            "num_dim = len(weight_floor), "
            "len(values) must be 2 ** num_dim, "
            f"But len(weight_floor) = {len(weight_floor)}, "
            f"len(values) = {len(values)}"
        )

    if len(weight_floor) == 1:  # one dimension
        return values[0] * weight_floor[0] + values[1] * weight_ceil[0]
    # multi dimension
    values_floor = pyramid_combination(
        values=values[::2],
        weight_floor=weight_floor[:-1],
        weight_ceil=weight_ceil[:-1],
    )
    values_floor = values_floor * weight_floor[-1]
    values_ceil = pyramid_combination(
        values=values[1::2],
        weight_floor=weight_floor[:-1],
        weight_ceil=weight_ceil[:-1],
    )
    values_ceil = values_ceil * weight_ceil[-1]
    return values_floor + values_ceil


def resample(
    vol: tf.Tensor,
    loc: tf.Tensor,
    interpolation: str = "linear",
    zero_boundary: bool = True,
) -> tf.Tensor:
    r"""
    Sample the volume at given locations.

    Input has

    - volume, vol, of shape = (batch, v_dim 1, ..., v_dim n),
      or (batch, v_dim 1, ..., v_dim n, ch),
      where n is the dimension of volume,
      ch is the extra dimension as features.

      Denote vol_shape = (v_dim 1, ..., v_dim n)

    - location, loc, of shape = (batch, l_dim 1, ..., l_dim m, n),
      where m is the dimension of output.

      Denote loc_shape = (l_dim 1, ..., l_dim m)

    Reference:

    - neuron's interpn
      https://github.com/adalca/neurite/blob/legacy/neuron/utils.py

      Difference

      1. they dont have batch size
      2. they support more dimensions in vol

      TODO try not using stack as neuron claims it's slower

    :param vol: shape = (batch, \*vol_shape) or (batch, \*vol_shape, ch)
      with the last channel for features
    :param loc: shape = (batch, \*loc_shape, n)
      such that loc[b, l1, ..., lm, :] = [v1, ..., vn] is of shape (n,),
      which represents a point in vol, with coordinates (v1, ..., vn)
    :param interpolation: linear only, TODO support nearest
    :param zero_boundary: if true, values on or outside boundary will be zeros
    :return: shape = (batch, \*loc_shape) or (batch, \*loc_shape, ch)
    """

    if interpolation != "linear":
        raise ValueError("resample supports only linear interpolation")

    # init
    batch_size = vol.shape[0]
    loc_shape = loc.shape[1:-1]
    dim_vol = loc.shape[-1]  # dimension of vol, n
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

    # get floor/ceil for loc and stack, then clip together
    # loc, loc_floor, loc_ceil are have shape (batch, *loc_shape, n)
    loc_ceil = tf.math.ceil(loc)
    loc_floor = loc_ceil - 1
    # (batch, *loc_shape, n, 3)
    clipped = tf.stack([loc, loc_floor, loc_ceil], axis=-1)
    clip_value_max = tf.cast(vol_shape, dtype=clipped.dtype) - 1  # (n,)
    clipped_shape = [1] * (len(loc_shape) + 1) + [dim_vol, 1]
    clip_value_max = tf.reshape(clip_value_max, shape=clipped_shape)
    clipped = tf.clip_by_value(clipped, clip_value_min=0, clip_value_max=clip_value_max)

    # loc_floor_ceil has n sublists
    # each one corresponds to the floor and ceil coordinates for d-th dimension
    # each tensor is of shape (batch, *loc_shape), dtype int32

    # weight_floor has n tensors
    # each tensor is the weight for the corner of floor coordinates
    # each tensor's shape is (batch, *loc_shape) if volume has no feature channel
    #                        (batch, *loc_shape, 1) if volume has feature channel
    loc_floor_ceil, weight_floor, weight_ceil = [], [], []
    # using for loop is faster than using list comprehension
    for dim in range(dim_vol):
        # shape = (batch, *loc_shape)
        c_clipped = clipped[..., dim, 0]
        c_floor = clipped[..., dim, 1]
        c_ceil = clipped[..., dim, 2]
        w_floor = c_ceil - c_clipped  # shape = (batch, *loc_shape)
        w_ceil = c_clipped - c_floor if zero_boundary else 1 - w_floor
        if has_ch:
            w_floor = tf.expand_dims(w_floor, -1)  # shape = (batch, *loc_shape, 1)
            w_ceil = tf.expand_dims(w_ceil, -1)  # shape = (batch, *loc_shape, 1)

        loc_floor_ceil.append([tf.cast(c_floor, tf.int32), tf.cast(c_ceil, tf.int32)])
        weight_floor.append(w_floor)
        weight_ceil.append(w_ceil)

    # 2**n corners, each is a list of n binary values
    corner_indices = get_n_bits_combinations(num_bits=len(vol_shape))

    # batch_coords[b, l1, ..., lm] = b
    # range(batch_size) on axis 0 and repeated on other axes
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
    sampled = pyramid_combination(
        values=corner_values, weight_floor=weight_floor, weight_ceil=weight_ceil
    )
    return sampled


def gen_rand_affine_transform(
    batch_size: int, scale: float, seed: (int, None) = None
) -> tf.Tensor:
    """
    Function that generates a random 3D transformation parameters for a batch of data.

    for 3D coordinates, affine transformation is

    .. code-block:: text

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

    .. code-block:: text

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

    In our case,

    - a = old
    - b = new
    - x = T

    To generate random transformation,
    we choose to add random perturbation to corner coordinates as follows:
    for corner of coordinates (x, y, z), the noise is

        -(x, y, z) .* (r1, r2, r3)

    where ri is a random number between (0, scale).
    So

        (x', y', z') = (x, y, z) .* (1-r1, 1-r2, 1-r3)

    Thus, we can directly sample between 1-scale and 1 instead

    We choose to calculate the transformation based on
    four corners in a cube centered at (0, 0, 0).
    A cube is shown as below, where

    - C = (-1, -1, -1)
    - G = (-1, -1, 1)
    - D = (-1, 1, -1)
    - A = (1, -1, -1)

    .. code-block:: text

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

    :param batch_size: int
    :param scale: a float number between 0 and 1
    :param seed: control the randomness
    :return: shape = (batch, 4, 3)
    """

    assert 0 <= scale <= 1
    np.random.seed(seed)
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


def gen_rand_ddf(
    batch_size: int,
    image_size: tuple,
    field_strength: (tuple, list),
    low_res_size: (tuple, list),
    seed: (int, None) = None,
) -> tf.Tensor:
    """
    Function that generates a random 3D DDF for a batch of data.

    :param batch_size:
    :param image_size:
    :param field_strength: maximum field strength, computed as a U[0,field_strength]
    :param low_res_size: low_resolution deformation field that will be upsampled to
        the original size in order to get smooth and more realistic fields.
    :param seed: control the randomness
    :return:
    """

    np.random.seed(seed)
    low_res_strength = np.random.uniform(0, field_strength, (batch_size, 1, 1, 1, 3))
    low_res_field = low_res_strength * np.random.randn(
        batch_size, low_res_size[0], low_res_size[1], low_res_size[2], 3
    )
    high_res_field = resize3d(low_res_field, image_size)
    return high_res_field


def warp_grid(grid: tf.Tensor, theta: tf.Tensor) -> tf.Tensor:
    """
    Perform transformation on the grid.

    - grid_padded[i,j,k,:] = [i j k 1]
    - grid_warped[b,i,j,k,p] = sum_over_q (grid_padded[i,j,k,q] * theta[b,q,p])

    :param grid: shape = (dim1, dim2, dim3, 3), grid[i,j,k,:] = [i j k]
    :param theta: parameters of transformation, shape = (batch, 4, 3)
    :return: shape = (batch, dim1, dim2, dim3, 3)
    """

    # grid_padded[i,j,k,:] = [i j k 1], shape = (dim1, dim2, dim3, 4)
    grid_padded = tf.concat([grid, tf.ones_like(grid[..., :1])], axis=3)

    # grid_warped[b,i,j,k,p] = sum_over_q (grid_padded[i,j,k,q] * theta[b,q,p])
    # shape = (batch, dim1, dim2, dim3, 3)
    grid_warped = tf.einsum("ijkq,bqp->bijkp", grid_padded, theta)
    return grid_warped


def warp_image_ddf(
    image: tf.Tensor, ddf: tf.Tensor, grid_ref: (tf.Tensor, None)
) -> tf.Tensor:
    """
    Warp an image with given DDF.

    :param image: an image to be warped, shape = (batch, m_dim1, m_dim2, m_dim3)
        or (batch, m_dim1, m_dim2, m_dim3, ch)
    :param ddf: shape = (batch, f_dim1, f_dim2, f_dim3, 3)
    :param grid_ref: shape = (f_dim1, f_dim2, f_dim3, 3)
        or (1, f_dim1, f_dim2, f_dim3, 3)
        if None grid_reg will be calculated based on ddf
    :return: shape = (batch, f_dim1, f_dim2, f_dim3)
        or (batch, f_dim1, f_dim2, f_dim3, ch)
    """
    if len(image.shape) not in [4, 5]:
        raise ValueError(
            f"image shape must be (batch, m_dim1, m_dim2, m_dim3) "
            f"or (batch, m_dim1, m_dim2, m_dim3, ch),"
            f" got {image.shape}"
        )
    if not (len(ddf.shape) == 5 and ddf.shape[-1] == 3):
        raise ValueError(
            f"ddf shape must be (batch, f_dim1, f_dim2, f_dim3, 3), got {ddf.shape}"
        )

    if grid_ref is None:
        # shape = (f_dim1, f_dim2, f_dim3, 3)
        grid_ref = get_reference_grid(grid_size=ddf.shape[1:4])
    if len(grid_ref.shape) == 4:
        # shape = (1, f_dim1, f_dim2, f_dim3, 3)
        grid_ref = grid_ref[None, ...]
    if not (
        len(grid_ref.shape) == 5
        and grid_ref.shape[0] == 1
        and grid_ref.shape[2:] == ddf.shape[2:]
    ):
        raise ValueError(
            f"grid_ref shape must be (1, f_dim1, f_dim2, f_dim3, 3) or None, "
            f"(f_dim1, f_dim2, f_dim3) must be the same as ddf, "
            f"got grid_ref.shape = {grid_ref.shape} and ddf.shape = {ddf.shape}."
        )

    return resample(vol=image, loc=grid_ref + ddf)


def resize3d(
    image: tf.Tensor, size: (tuple, list), method: str = tf.image.ResizeMethod.BILINEAR
) -> tf.Tensor:
    """
    Tensorflow does not have resize 3d, therefore the resize is performed two folds.

    - resize dim2 and dim3
    - resize dim1 and dim2

    :param image: tensor of shape = (batch, dim1, dim2, dim3, channels)
                                 or (batch, dim1, dim2, dim3)
                                 or (dim1, dim2, dim3)
    :param size: tuple, (out_dim1, out_dim2, out_dim3)
    :param method: str, one of tf.image.ResizeMethod
    :return: tensor of shape = (batch, out_dim1, out_dim2, out_dim3, channels)
                            or (batch, dim1, dim2, dim3)
                            or (dim1, dim2, dim3)
    """
    # sanity check
    image_dim = len(image.shape)
    if image_dim not in [3, 4, 5]:
        raise ValueError(
            "resize3d takes input image of dimension 3 or 4 or 5,"
            "corresponding to (dim1, dim2, dim3) "
            "or (batch, dim1, dim2, dim3)"
            "or (batch, dim1, dim2, dim3, channels),"
            "got image shape{}".format(image.shape)
        )
    if len(size) != 3:
        raise ValueError("resize3d takes size of type tuple/list and of length 3")

    # init
    if image_dim == 5:
        has_channel = True
        has_batch = True
        input_image_shape = image.shape[1:4]
    elif image_dim == 4:
        has_channel = False
        has_batch = True
        input_image_shape = image.shape[1:4]
    else:
        has_channel = False
        has_batch = False
        input_image_shape = image.shape[0:3]

    # no need of resize
    if input_image_shape == tuple(size):
        return image

    # expand to five dimensions
    if not has_batch:
        image = tf.expand_dims(image, axis=0)
    if not has_channel:
        image = tf.expand_dims(image, axis=-1)
    assert len(image.shape) == 5  # (batch, dim1, dim2, dim3, channels)
    image_shape = tf.shape(image)

    # merge axis 0 and 1
    output = tf.reshape(
        image, (-1, image_shape[2], image_shape[3], image_shape[4])
    )  # (batch * dim1, dim2, dim3, channels)

    # resize dim2 and dim3
    output = tf.image.resize(
        images=output, size=size[1:], method=method
    )  # (batch * dim1, out_dim2, out_dim3, channels)

    # split axis 0 and merge axis 3 and 4
    output = tf.reshape(
        output, shape=(-1, image_shape[1], size[1], size[2] * image_shape[4])
    )  # (batch, dim1, out_dim2, out_dim3 * channels)

    # resize dim1 and dim2
    output = tf.image.resize(
        images=output, size=size[:2], method=method
    )  # (batch, out_dim1, out_dim2, out_dim3 * channels)

    # reshape
    output = tf.reshape(
        output, shape=[-1, *size, image_shape[4]]
    )  # (batch, out_dim1, out_dim2, out_dim3, channels)

    # squeeze to original dimension
    if not has_batch:
        output = tf.squeeze(output, axis=0)
    if not has_channel:
        output = tf.squeeze(output, axis=-1)
    return output


def gaussian_filter_3d(kernel_sigma: (list, tuple, int)) -> tf.Tensor:
    """
    Define a gaussian filter in 3d for smoothing.

    The filter size is defined 3*kernel_sigma


    :param kernel_sigma: the deviation at each direction (list)
        or use an isotropic deviation (int)
    :return: kernel: tf.Tensor specify a gaussian kernel of shape:
        [3*k for k in kernel_sigma]
    """
    if isinstance(kernel_sigma, int):
        kernel_sigma = (kernel_sigma, kernel_sigma, kernel_sigma)

    kernel_size = [
        int(np.ceil(ks * 3) + np.mod(np.ceil(ks * 3) + 1, 2)) for ks in kernel_sigma
    ]

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    coord = [np.arange(ks) for ks in kernel_size]

    xx, yy, zz = np.meshgrid(coord[0], coord[1], coord[2], indexing="ij")
    xyz_grid = np.concatenate(
        (xx[np.newaxis], yy[np.newaxis], zz[np.newaxis]), axis=0
    )  # 2, y, x

    mean = np.asarray([(ks - 1) / 2.0 for ks in kernel_size])
    mean = mean.reshape(-1, 1, 1, 1)
    variance = np.asarray([ks ** 2.0 for ks in kernel_sigma])
    variance = variance.reshape(-1, 1, 1, 1)

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    # 2.506628274631 = sqrt(2 * pi)

    norm_kernel = 1.0 / (np.sqrt(2 * np.pi) ** 3 + np.prod(kernel_sigma))
    kernel = norm_kernel * np.exp(
        -np.sum((xyz_grid - mean) ** 2.0 / (2 * variance), axis=0)
    )

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / np.sum(kernel)

    # Reshape
    kernel = kernel.reshape(kernel_size[0], kernel_size[1], kernel_size[2])

    # Total kernel
    total_kernel = np.zeros(tuple(kernel_size) + (3, 3))
    total_kernel[..., 0, 0] = kernel
    total_kernel[..., 1, 1] = kernel
    total_kernel[..., 2, 2] = kernel

    return tf.convert_to_tensor(total_kernel, dtype=tf.float32)
