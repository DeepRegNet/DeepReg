import tensorflow as tf

EPS = 1.0e-6  # epsilon to prevent NaN

"""
similarity
"""


def get_similarity_fn(config):
    """
    :param config:
    :return:
    """
    if config["name"] == "multi_scale":

        def loss(y_true, y_pred):
            return tf.reduce_mean(
                multi_scale_loss(
                    y_true=y_true, y_pred=y_pred, **config["multi_scale"]
                )
            )  # [batch]

        return loss
    elif config["name"] == "single_scale":

        def loss(y_true, y_pred):
            return tf.reduce_mean(
                single_scale_loss(
                    y_true=y_true, y_pred=y_pred, **config["single_scale"]
                )
            )  # [batch]

        return loss
    else:
        raise ValueError("Unknown loss type.")


def multi_scale_loss(y_true, y_pred, loss_type, loss_scales):
    """
    apply the loss at different scales (gaussian smoothing)
    assuming values are between 0 and 1
    :param y_true: shape = [batch, dim1, dim2, dim3]
    :param y_pred: shape = [batch, dim1, dim2, dim3]
    :param loss_type:
    :param loss_scales:
    :return: [batch]
    """
    assert len(y_true.shape) == 4
    assert len(y_pred.shape) == 4
    label_loss_all = tf.stack(
        [
            single_scale_loss(
                y_true=separable_filter3d(y_true, gauss_kernel1d(s)),
                y_pred=separable_filter3d(y_pred, gauss_kernel1d(s)),
                loss_type=loss_type,
            )
            for s in loss_scales
        ],
        axis=1,
    )
    return tf.reduce_mean(label_loss_all, axis=1)


def single_scale_loss(y_true, y_pred, loss_type):
    """

    :param y_true: shape = [batch, dim1, dim2, dim3]
    :param y_pred: shape = [batch, dim1, dim2, dim3]
    :param loss_type:
    :return: shape = [batch]
    """
    if loss_type == "cross-entropy":
        return weighted_binary_cross_entropy(y_true, y_pred)
    elif loss_type == "mean-squared":
        return tf.reduce_mean(
            tf.math.squared_difference(y_true, y_pred), axis=[1, 2, 3]
        )
    elif loss_type == "dice":
        return 1 - dice_score(y_true, y_pred)
    elif loss_type == "dice_generalized":
        return 1 - dice_score_generalized(y_true, y_pred)
    elif loss_type == "jaccard":
        return 1 - jaccard_index(y_true, y_pred)
    else:
        raise ValueError("Unknown loss type.")


def weighted_binary_cross_entropy(y_true, y_pred, pos_weight=1):
    """
    loss = − pos_w * y_true log(y_pred) - (1−y_true) log(1−y_pred)
    :param y_true: shape = [batch, dim1, dim2, dim3]
    :param y_pred: shape = [batch, dim1, dim2, dim3]
    :param pos_weight: weight of positive class, scalar
    :return: shape = [batch]
    """
    y_pred = tf.clip_by_value(y_pred, EPS, 1 - EPS)
    return -pos_weight * tf.reduce_mean(
        y_true * tf.math.log(y_pred), axis=[1, 2, 3]
    ) - tf.reduce_mean((1 - y_true) * tf.math.log(1 - y_pred), axis=[1, 2, 3])


def dice_score(y_true, y_pred, binary=False):
    """
    num = 2 * y_true * y_pred
    denom = y_true + y_pred
    dice score = num / denom
    num and denom are summed over the entire image first
    :param y_true: shape = [batch, dim1, dim2, dim3]
    :param y_pred: shape = [batch, dim1, dim2, dim3]
    :param binary: true if the y should be projected to 0 or 1
    :return: shape = [batch]
    """
    if binary:
        y_true = tf.cast(y_true >= 0.5, dtype=tf.float32)
        y_pred = tf.cast(y_pred >= 0.5, dtype=tf.float32)
    numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3]) * 2 + EPS
    denominator = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        + EPS
    )
    return numerator / denominator


def dice_score_generalized(y_true, y_pred, pos_weight=1, neg_weight=0):
    """
    let y_prod = y_true * y_pred
        y_sum  = y_true + y_pred
    num = 2 *  (pos_w * y_true * y_pred + neg_w * (1−y_true) * (1−y_pred))
        = 2 *  ((pos_w+neg_w) * y_prod - neg_w * y_sum + neg_w)
    denom = (pos_w * (y_true + y_pred) + neg_w * (1−y_true + 1−y_pred))
          = (pos_w-neg_w) * y_sum + 2 * neg_w
    dice score = num / denom
    num and denom are summed over the entire image first
    :param y_true: shape = [batch, dim1, dim2, dim3]
    :param y_pred: shape = [batch, dim1, dim2, dim3]
    :param pos_weight:
    :param neg_weight:
    :return: shape = [batch]
    """
    y_prod = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    y_sum = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(
        y_pred, axis=[1, 2, 3]
    )

    numerator = (
        2
        * (
            (pos_weight + neg_weight) * y_prod
            - neg_weight * y_sum
            + neg_weight
        )
        + EPS
    )
    denominator = (pos_weight - neg_weight) * y_sum + 2 * neg_weight + EPS
    return numerator / denominator


def jaccard_index(y_true, y_pred):
    """
    num = y_true * y_pred
    denom = y_true + y_pred - y_true * y_pred
    jaccard index = num / denom
    :param y_true: shape = [batch, dim1, dim2, dim3]
    :param y_pred: shape = [batch, dim1, dim2, dim3]
    :return: shape = [batch]
    """
    numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3]) + EPS
    denominator = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        - numerator
        + EPS
    )
    return numerator / denominator


def gauss_kernel1d(sigma):
    if sigma == 0:
        return tf.constant(0)
    else:
        tail = int(sigma * 3)
        k = tf.exp(
            [-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)]
        )
        return k / tf.reduce_sum(k)


def cauchy_kernel1d(sigma):  # this is an approximation
    if sigma == 0:
        return 0
    else:
        tail = int(sigma * 5)
        k = tf.math.reciprocal(
            [((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)]
        )
        return k / tf.reduce_sum(k)


def separable_filter3d(x, kernel):
    """
    :param x: shape = [batch, dim1, dim2, dim3]
    :param kernel:
    :return: shape = [batch, dim1, dim2, dim3]
    """
    if len(kernel.shape) == 0:
        return x
    else:
        strides = [1, 1, 1, 1, 1]
        x = tf.nn.conv3d(
            tf.nn.conv3d(
                tf.nn.conv3d(
                    tf.expand_dims(x, axis=4),
                    filters=tf.reshape(kernel, [-1, 1, 1, 1, 1]),
                    strides=strides,
                    padding="SAME",
                ),
                filters=tf.reshape(kernel, [1, -1, 1, 1, 1]),
                strides=strides,
                padding="SAME",
            ),
            filters=tf.reshape(kernel, [1, 1, -1, 1, 1]),
            strides=strides,
            padding="SAME",
        )
        return x[:, :, :, :, 0]


"""
distance
"""


def compute_centroid(mask, grid):
    """
    calculate the centroid of the mask
    :param mask: shape = [batch, dim1, dim2, dim3]
    :param grid: shape = [dim1, dim2, dim3, 3]
    :return: shape = [batch, 3]
    """
    bool_mask = tf.expand_dims(
        tf.cast(mask >= 0.5, dtype=tf.float32), axis=4
    )  # [batch, dim1, dim2, dim3, 1]
    masked_grid = bool_mask * tf.expand_dims(
        grid, axis=0
    )  # [batch, dim1, dim2, dim3, 3]
    numerator = tf.reduce_sum(masked_grid, axis=[1, 2, 3]) + EPS  # [batch, 3]
    denominator = tf.reduce_sum(bool_mask, axis=[1, 2, 3]) + EPS  # [batch, 1]
    return numerator / denominator  # [batch, 3]


def compute_centroid_distance(y_true, y_pred, grid):
    """
    :param y_true: shape = [batch, dim1, dim2, dim3]
    :param y_pred: shape = [batch, dim1, dim2, dim3]
    :param grid: shape = [dim1, dim2, dim3, 3]
    :return: shape = [batch]
    """
    c1 = compute_centroid(mask=y_pred, grid=grid)  # shape = [batch, 3]
    c2 = compute_centroid(mask=y_true, grid=grid)  # shape = [batch, 3]
    return tf.sqrt(tf.reduce_sum((c1 - c2) ** 2, axis=[1]))
