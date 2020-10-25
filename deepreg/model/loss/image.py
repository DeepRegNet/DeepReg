"""
Module provides different loss functions for calculating the dissimilarities between images.
"""
import tensorflow as tf

EPS = tf.keras.backend.epsilon()


def dissimilarity_fn(
    y_true: tf.Tensor, y_pred: tf.Tensor, name: str, **kwargs
) -> tf.Tensor:
    """
    Returns the calculated dissimilarity for each batch.

    :param y_true: fixed_image, shape = (batch, f_dim1, f_dim2, f_dim3)
    :param y_pred: warped_moving_image, shape = (batch, f_dim1, f_dim2, f_dim3)
    :param name: name of the dissimilarity function
    :param kwargs: absorb additional parameters
    :return: shape = (batch,)
    """

    # shape = (batch, f_dim1, f_dim2, f_dim3, 1)
    y_true = tf.expand_dims(y_true, axis=4)
    y_pred = tf.expand_dims(y_pred, axis=4)
    if name == "lncc":
        return -local_normalized_cross_correlation(y_true, y_pred, **kwargs)
    elif name == "ssd":
        return ssd(y_true, y_pred)
    elif name == "gmi":
        return -global_mutual_information(y_true, y_pred)
    else:
        raise ValueError(
            f"Unknown loss type {name}, supported losses are lncc, ssd and gmi."
        )


def local_normalized_cross_correlation(
    y_true: tf.Tensor,
    y_pred: tf.Tensor,
    kernel_size: int = 9,
    kernel_type: str = "rectangular",
    **kwargs,
) -> tf.Tensor:
    """
    Local squared zero-normalized cross-correlation.
    The loss is based on a moving kernel/window over the y_true/y_pred,
    within the window the square of zncc is calculated.
    The kernel can be a rectangular / triangular / gaussian window.
    The final loss is the averaged loss over all windows.

    Reference:

    - Zero-normalized cross-correlation (ZNCC): https://en.wikipedia.org/wiki/Cross-correlation
    - Code: https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py

    :param y_true: shape = (batch, dim1, dim2, dim3, ch)
    :param y_pred: shape = (batch, dim1, dim2, dim3, ch)
    :param kernel_size: int. Kernel size or kernel sigma for kernel_type='gauss'.
    :param kernel_type: str ('triangular', 'gaussian' default: 'rectangular')
    :param kwargs: absorb additional parameters
    :return: shape = (batch,)
    """

    # [dim1, dim2, dim3, d_in, d_out]
    # ch must be evenly divisible by d_in
    ch = y_true.shape[4]

    if kernel_type == "rectangular":
        filters = tf.ones(shape=[kernel_size, kernel_size, kernel_size, ch, 1])
        kernel_vol = kernel_size ** 3

    elif kernel_type == "triangular":
        fsize = int((kernel_size + 1) / 2)
        pad_filter = tf.constant(
            [
                [0, 0],
                [int((fsize - 1) / 2), int((fsize - 1) / 2)],
                [int((fsize - 1) / 2), int((fsize - 1) / 2)],
                [int((fsize - 1) / 2), int((fsize - 1) / 2)],
                [0, 0],
            ]
        )

        f1 = tf.ones(shape=(1, fsize, fsize, fsize, 1)) / fsize
        f1 = tf.pad(f1, pad_filter, "CONSTANT")
        f2 = tf.ones(shape=(fsize, fsize, fsize, 1, ch)) / fsize

        filters = tf.nn.conv3d(f1, f2, strides=[1, 1, 1, 1, 1], padding="SAME")
        filters = tf.transpose(filters, perm=[1, 2, 3, 4, 0])
        kernel_vol = tf.reduce_sum(filters ** 2)

    elif kernel_type == "gaussian":
        mean = (kernel_size - 1) / 2.0
        sigma = kernel_size / 3

        grid_dim = tf.range(0, kernel_size)
        grid_dim_ch = tf.range(0, ch)
        grid = tf.expand_dims(
            tf.cast(
                tf.stack(tf.meshgrid(grid_dim, grid_dim, grid_dim, grid_dim_ch), 0),
                dtype="float32",
            ),
            axis=-1,
        )
        filters = tf.exp(
            -tf.reduce_sum(tf.square(grid - mean), axis=0) / (2 * sigma ** 2)
        )
        kernel_vol = tf.reduce_sum(filters ** 2)

    else:
        raise ValueError(
            f"Wrong kernel_type for LNCC loss type. "
            f"Please, specify a valid type 'rectangular' / 'triangular' / 'gaussian',"
            f"got {kernel_type}"
        )

    strides = [1, 1, 1, 1, 1]
    padding = "SAME"

    # t = y_true, p = y_pred
    # (batch, dim1, dim2, dim3, ch)
    t2 = y_true * y_true
    p2 = y_pred * y_pred
    tp = y_true * y_pred

    # sum over kernel
    # (batch, dim1, dim2, dim3, 1)
    t_sum = tf.nn.conv3d(y_true, filters=filters, strides=strides, padding=padding)
    p_sum = tf.nn.conv3d(y_pred, filters=filters, strides=strides, padding=padding)
    t2_sum = tf.nn.conv3d(t2, filters=filters, strides=strides, padding=padding)
    p2_sum = tf.nn.conv3d(p2, filters=filters, strides=strides, padding=padding)
    tp_sum = tf.nn.conv3d(tp, filters=filters, strides=strides, padding=padding)

    # average over kernel
    # (batch, dim1, dim2, dim3, 1)
    t_avg = t_sum / kernel_vol
    p_avg = p_sum / kernel_vol

    # normalized cross correlation between t and p
    # sum[(t - mean[t]) * (p - mean[p])] / std[t] / std[p]
    # denoted by num / denom
    # assume we sum over N values
    # num = sum[t * p - mean[t] * p - t * mean[p] + mean[t] * mean[p]]
    #     = sum[t*p] - sum[t] * sum[p] / N * 2 + sum[t] * sum[p] / N
    #     = sum[t*p] - sum[t] * sum[p] / N
    #     = sum[t*p] - sum[t] * mean[p] = cross
    # the following is actually squared ncc
    # shape = (batch, dim1, dim2, dim3, 1)
    cross = tp_sum - p_avg * t_sum
    t_var = t2_sum - t_avg * t_sum  # std[t] ** 2
    p_var = p2_sum - p_avg * p_sum  # std[p] ** 2
    ncc = (cross * cross + EPS) / (t_var * p_var + EPS)
    return tf.reduce_mean(ncc, axis=[1, 2, 3, 4])


def ssd(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Sum of squared distance between y_true and y_pred.

    :param y_true: shape = (batch, dim1, dim2, dim3, ch)
    :param y_pred: shape = (batch, dim1, dim2, dim3, ch)
    :return: shape = (batch,)
    """
    return tf.reduce_mean(tf.square(y_true - y_pred), axis=[1, 2, 3, 4])


def global_mutual_information(
    y_true: tf.Tensor, y_pred: tf.Tensor, num_bins: int = 23, sigma_ratio: float = 0.5
) -> tf.Tensor:
    """
    Differentiable global mutual information loss via Parzen windowing method.

    Reference: https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5, Algorithm 1

    :param y_true: shape = (batch, dim1, dim2, dim3, ch)
    :param y_pred: shape = (batch, dim1, dim2, dim3, ch)
    :param num_bins: int, number of bins for intensity
    :param sigma_ratio: float, a hyper param for gaussian function
    :return: shape = (batch,)
    """

    # intensity is split into bins between 0, 1
    y_true = tf.clip_by_value(y_true, 0, 1)
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    bin_centers = tf.linspace(0.0, 1.0, num_bins)  # (num_bins,)
    bin_centers = bin_centers[None, None, ...]  # (1, 1, num_bins)
    sigma = (
        tf.reduce_mean(bin_centers[:, :, 1:] - bin_centers[:, :, :-1]) * sigma_ratio
    )  # scalar, sigma in the Gaussian function (weighting function W)
    preterm = 1 / (2 * tf.math.square(sigma))  # scalar
    batch, w, h, z, c = y_true.shape
    y_true = tf.reshape(y_true, [batch, w * h * z * c, 1])  # (batch, nb_voxels, 1)
    y_pred = tf.reshape(y_pred, [batch, w * h * z * c, 1])  # (batch, nb_voxels, 1)
    nb_voxels = y_true.shape[1] * 1.0  # w * h * z, number of voxels

    # each voxel contributes continuously to a range of histogram bin
    Ia = tf.math.exp(
        -preterm * tf.math.square(y_true - bin_centers)
    )  # (batch, nb_voxels, num_bins)
    Ia /= tf.reduce_sum(Ia, -1, keepdims=True)  # (batch, nb_voxels, num_bins)
    Ia = tf.transpose(Ia, (0, 2, 1))  # (batch, num_bins, nb_voxels)
    pa = tf.reduce_mean(Ia, axis=-1, keepdims=True)  # (batch, num_bins, 1)

    Ib = tf.math.exp(
        -preterm * tf.math.square(y_pred - bin_centers)
    )  # (batch, nb_voxels, num_bins)
    Ib /= tf.reduce_sum(Ib, -1, keepdims=True)  # (batch, nb_voxels, num_bins)
    pb = tf.reduce_mean(Ib, axis=1, keepdims=True)  # (batch, 1, num_bins)

    papb = tf.matmul(pa, pb)  # (batch, num_bins, num_bins)
    pab = tf.matmul(Ia, Ib)  # (batch, num_bins, num_bins)
    pab /= nb_voxels

    # MI: sum(P_ab * log(P_ab/P_aP_b))
    div = (pab + EPS) / (papb + EPS)
    return tf.reduce_sum(pab * tf.math.log(div + EPS), axis=[1, 2])
