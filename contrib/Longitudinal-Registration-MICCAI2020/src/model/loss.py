import numpy as np
import tensorflow as tf


class NCC:  # added new
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None, eps=1e-5):
        self.win = win
        self.eps = eps

    def ncc(self, I, J):
        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = 3

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        print(I.shape, J.shape, "ndims:", ndims, "conv%dd" % ndims)
        conv_fn = tf.nn.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        sum_filt = tf.ones([*self.win, 1, 1])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = "SAME"

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)
        print("here")

        # return negative cc.
        return tf.reduce_mean(cc)

    def loss(self, I, J):
        return -self.ncc(I, J)


def single_scale_loss(label_fixed, label_moving, loss_type):
    if loss_type == "cross-entropy":
        label_loss_batch = tf.reduce_mean(
            weighted_binary_cross_entropy(label_fixed, label_moving), axis=[1, 2, 3, 4]
        )
    elif loss_type == "mean-squared":
        label_loss_batch = tf.reduce_mean(
            tf.math.squared_difference(label_fixed, label_moving), axis=[1, 2, 3, 4]
        )
    elif loss_type == "ssd":  # sum-square-difference
        label_loss_batch = tf.math.squared_difference(label_fixed, label_moving)
    elif loss_type == "dice":
        if len(label_fixed.shape) == 4:
            label_fixed = tf.expand_dims(label_fixed, axis=4)
        if len(label_moving.shape) == 4:
            label_moving = tf.expand_dims(label_moving, axis=4)
        label_loss_batch = 1 - dice_simple(label_fixed, label_moving)
    elif loss_type == "jaccard":
        label_loss_batch = 1 - jaccard_simple(label_fixed, label_moving)
    else:
        raise ValueError("Unknown loss type.")
    return label_loss_batch


def multi_scale_loss(label_fixed, label_moving, loss_type, loss_scales):
    # TODO change to one conv?
    label_loss_all = tf.stack(
        [
            single_scale_loss(
                separable_filter3d(label_fixed, gauss_kernel1d(s)),
                separable_filter3d(label_moving, gauss_kernel1d(s)),
                loss_type,
            )
            for s in loss_scales
        ],
        axis=1,
    )
    return tf.reduce_mean(label_loss_all, axis=1)


def weighted_binary_cross_entropy(ts, ps, pw=1, eps=1e-6):
    ps = tf.clip_by_value(ps, eps, 1 - eps)
    return -tf.reduce_sum(
        tf.concat([ts * pw, 1 - ts], axis=4)
        * tf.math.log(tf.concat([ps, 1 - ps], axis=4)),
        axis=4,
        keep_dims=True,
    )


def dice_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts * ps, axis=[1, 2, 3, 4]) * 2
    denominator = (
        tf.reduce_sum(ts, axis=[1, 2, 3, 4])
        + tf.reduce_sum(ps, axis=[1, 2, 3, 4])
        + eps_vol
    )
    return numerator / denominator


def dice_generalised(ts, ps, weights):
    ts2 = tf.concat([ts, 1 - ts], axis=4)
    ps2 = tf.concat([ps, 1 - ps], axis=4)
    numerator = 2 * tf.reduce_sum(
        tf.reduce_sum(ts2 * ps2, axis=[1, 2, 3]) * weights, axis=1
    )
    denominator = tf.reduce_sum(
        (tf.reduce_sum(ts2, axis=[1, 2, 3]) + tf.reduce_sum(ps2, axis=[1, 2, 3]))
        * weights,
        axis=1,
    )
    return numerator / denominator


def jaccard_simple(ts, ps, eps_vol=1e-6):
    numerator = tf.reduce_sum(ts * ps, axis=[1, 2, 3, 4])
    denominator = (
        tf.reduce_sum(tf.square(ts), axis=[1, 2, 3, 4])
        + tf.reduce_sum(tf.square(ps), axis=[1, 2, 3, 4])
        - numerator
        + eps_vol
    )
    return numerator / denominator


def gauss_kernel1d(sigma):
    if sigma == 0:
        return tf.constant(0)
    else:
        tail = int(sigma * 3)
        k = tf.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)


def cauchy_kernel1d(sigma):  # this is an approximation
    if sigma == 0:
        return 0
    else:
        tail = int(sigma * 5)
        # k = tf.math.reciprocal(([((x/sigma)**2+1)*sigma*3.141592653589793 for x in range(-tail, tail+1)]))
        k = tf.math.reciprocal([((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)


def separable_filter3d(vol, kernel):
    if len(kernel.shape) == 0:
        return vol
    else:
        strides = [1, 1, 1, 1, 1]
        # TODO use keras for conv3d
        return tf.nn.conv3d(
            tf.nn.conv3d(
                tf.nn.conv3d(
                    vol, tf.reshape(kernel, [-1, 1, 1, 1, 1]), strides, "SAME"
                ),
                tf.reshape(kernel, [1, -1, 1, 1, 1]),
                strides,
                "SAME",
            ),
            tf.reshape(kernel, [1, 1, -1, 1, 1]),
            strides,
            "SAME",
        )


def local_displacement_energy(ddf, energy_type, energy_weight):
    def gradient_dx(fv):
        return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2

    def gradient_dy(fv):
        return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2

    def gradient_dz(fv):
        return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2

    def gradient_txyz(Txyz, fn):
        return tf.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], axis=4)

    def compute_gradient_norm(displacement, flag_l1=False):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        if flag_l1:
            norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
        else:
            norms = dTdx ** 2 + dTdy ** 2 + dTdz ** 2
        return tf.reduce_mean(norms, [1, 2, 3, 4])

    def compute_bending_energy(displacement):
        dTdx = gradient_txyz(displacement, gradient_dx)
        dTdy = gradient_txyz(displacement, gradient_dy)
        dTdz = gradient_txyz(displacement, gradient_dz)
        dTdxx = gradient_txyz(dTdx, gradient_dx)
        dTdyy = gradient_txyz(dTdy, gradient_dy)
        dTdzz = gradient_txyz(dTdz, gradient_dz)
        dTdxy = gradient_txyz(dTdx, gradient_dy)
        dTdyz = gradient_txyz(dTdy, gradient_dz)
        dTdxz = gradient_txyz(dTdx, gradient_dz)
        return tf.reduce_mean(
            dTdxx ** 2
            + dTdyy ** 2
            + dTdzz ** 2
            + 2 * dTdxy ** 2
            + 2 * dTdxz ** 2
            + 2 * dTdyz ** 2,
            [1, 2, 3, 4],
        )

    def bening_energy_4d(displacement):
        """displacement shape: """
        pass

    if energy_weight:
        if energy_type == "bending":
            energy = compute_bending_energy(ddf)
        elif energy_type == "gradient-l2":
            energy = compute_gradient_norm(ddf)
        elif energy_type == "gradient-l1":
            energy = compute_gradient_norm(ddf, flag_l1=True)
        else:
            raise ValueError("Unknown regularizer.")
    else:
        energy = tf.constant(0.0)

    return energy * energy_weight


def binary_dice(y_true, y_pred):
    if len(y_true.shape) == 4:
        y_true = tf.expand_dims(y_true, axis=4)
    if len(y_pred.shape) == 4:
        y_pred = tf.expand_dims(y_pred, axis=4)
    mask1 = y_true >= 0.5
    mask2 = y_pred >= 0.5
    vol1 = tf.reduce_sum(tf.cast(mask1, dtype=tf.float32), axis=[1, 2, 3, 4])
    vol2 = tf.reduce_sum(tf.cast(mask2, dtype=tf.float32), axis=[1, 2, 3, 4])
    dice = (
        tf.reduce_sum(tf.cast(mask1 & mask2, dtype=tf.float32), axis=[1, 2, 3, 4])
        * 2
        / (vol1 + vol2)
    )
    return dice


def compute_centroid(mask, grid, eps=1e-6):
    """

    :param mask: shape = [batch, dim1, dim2, dim3, 1]
    :param grid: shape = [dim1, dim2, dim3, 3]
    :return:
    """
    centroids = []
    for i in range(mask.shape[0]):
        bool_mask = mask[i, ..., 0] >= 0.5  # [dim1, dim2, dim3]
        masked = tf.boolean_mask(grid, bool_mask)  # [None, 3]
        centroid = tf.reduce_sum(masked, axis=0) / (
            tf.reduce_sum(tf.cast(bool_mask, tf.float32)) + eps
        )  # [3]
        centroids.append(centroid)
    return tf.stack(centroids, axis=0)

    # return tf.stack([tf.reduce_mean(tf.boolean_mask(grid, mask[i, ..., 0] >= 0.5), axis=0)
    #                  for i in range(mask.shape[0])], axis=0)


def compute_centroid_distance(y_true, y_pred, grid):
    """
    :param y_true: shape = [batch, dim1, dim2, dim3] or [batch, dim1, dim2, dim3, 1]
    :param y_pred: shape = [batch, dim1, dim2, dim3] or [batch, dim1, dim2, dim3, 1]
    :param grid: shape = [dim1, dim2, dim3, 3]
    :return:
    """
    if len(y_true.shape) == 4:
        y_true = tf.expand_dims(y_true, axis=4)
    if len(y_pred.shape) == 4:
        y_pred = tf.expand_dims(y_pred, axis=4)
    c1 = compute_centroid(y_pred, grid)
    c2 = compute_centroid(y_true, grid)
    return tf.sqrt(tf.reduce_sum((c1 - c2) ** 2))


# def loss_similarity_fn(y_true, y_pred):
#     """
#     add ncc loss
#     :param y_true: fixed_label, shape = [batch, f_dim1, f_dim2, f_dim3, (1)]
#     :param y_pred: warped_moving_label, shape = [batch, f_dim1, f_dim2, f_dim3, (1)]
#     :return:
#     """
#     if len(y_true.shape) == 4:
#         y_true = tf.expand_dims(y_true, axis=4)
#     if len(y_pred.shape) == 4:
#         y_pred = tf.expand_dims(y_pred, axis=4)
#     loss_similarity = tf.reduce_mean(multi_scale_loss(label_fixed=y_true,
#                                                       label_moving=y_pred,
#                                                       loss_type="dice",
#                                                       loss_scales=[0, 1, 2, 4, 8, 16, 32]))  # TODO move into config
#     return loss_similarity


def loss_ncc(y_true, y_pred):
    """normalized cross-correlation loss"""
    if len(y_true.shape) == 4:
        y_true = tf.expand_dims(y_true, axis=4)
    if len(y_pred.shape) == 4:
        y_pred = tf.expand_dims(y_pred, axis=4)

    loss_ncc = NCC().loss(y_true, y_pred)
    return loss_ncc


def loss_ssd(y_true, y_pred):
    """sum of squared error"""
    if len(y_true.shape) == 4:
        y_true = tf.expand_dims(y_true, axis=4)
    if len(y_pred.shape) == 4:
        y_pred = tf.expand_dims(y_pred, axis=4)
    return tf.math.squared_difference(y_true, y_pred)


def loss_mmd(x1, x2, sigmas=[1.0]):
    """the loss of maximum mean discrepancy."""
    x1 = tf.reshape(x1, [x1.shape[0], -1])
    x2 = tf.reshape(x2, [x2.shape[0], -1])
    sigmas = tf.constant(sigmas)
    diff = tf.reduce_mean(gaussian_kernel(x1, x1, sigmas))  # mean_x1x1
    diff -= 2 * tf.reduce_mean(gaussian_kernel(x1, x2, sigmas))  # mean_x1x2
    diff += tf.reduce_mean(gaussian_kernel(x2, x2, sigmas))  # mean_x2x2
    return diff


def gaussian_kernel(x1, x2, sigmas):
    beta = 1.0 / (2.0 * (tf.expand_dims(sigmas, 1)))
    dist = tf.reduce_sum(tf.square(tf.expand_dims(x1, 2) - tf.transpose(x2)), 1)
    dist = tf.transpose(dist)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))


def loss_global_mutual_information(y_true, y_pred):
    """
    differentiable global mutual information loss via Parzen windowing method.
    reference: https://dspace.mit.edu/handle/1721.1/123142

    :y_true: shape = (batch, dim1, dim2, dim3, ch)
    :y_pred: shape = (batch, dim1, dim2, dim3, ch)
    :return: shape = (batch,)
    """
    print(y_true.shape, y_pred.shape)
    if len(y_true.shape) == 4:
        y_true = tf.expand_dims(y_true, axis=4)
    if len(y_pred.shape) == 4:
        y_pred = tf.expand_dims(y_pred, axis=4)
    print(y_true.shape, y_pred.shape)
    bin_centers = tf.linspace(0.0, 1.1, 23)
    sigma_ratio = 0.5
    eps = 1.19209e-07

    sigma = tf.reduce_mean(bin_centers[1:] - bin_centers[:-1]) * sigma_ratio
    preterm = 1 / (2 * tf.math.square(sigma))

    batch, w, h, z, c = y_pred.shape
    y_true = tf.reshape(y_true, [batch, w * h * z, 1])
    y_pred = tf.reshape(y_pred, [batch, w * h * z, 1])
    nb_voxels = y_true.shape[1] * 1.0  # number of voxels

    vbc = bin_centers[None, None, ...]

    I_a = tf.math.exp(-preterm * tf.math.square(y_true - vbc))
    I_a /= tf.reduce_sum(I_a, -1, keepdims=True)
    pa = tf.reduce_mean(I_a, axis=1, keepdims=True)
    I_a_permute = tf.transpose(I_a, (0, 2, 1))

    I_b = tf.math.exp(-preterm * tf.math.square(y_pred - vbc))
    I_b /= tf.reduce_sum(I_b, -1, keepdims=True)
    pb = tf.reduce_mean(I_b, axis=1, keepdims=True)

    pa = tf.transpose(pa, (0, 2, 1))
    papb = tf.keras.backend.batch_dot(pa, pb, axes=(2, 1)) + eps
    pab = tf.keras.backend.batch_dot(I_a_permute, I_b, axes=(2, 1))
    pab /= nb_voxels

    return tf.reduce_sum(pab * tf.math.log(pab / papb + eps), axis=[1, 2])
