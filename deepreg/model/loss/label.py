"""
Module to perform label and prediction operations.
"""
import tensorflow as tf

EPS = 1.0e-6  # epsilon to prevent NaN


def get_similarity_fn(config):
    """
    Function to parse args from a config dictionary
    and return the loss by averaging batch loss returned by
    multi- or single-scale loss functions.

    :param config: dict, containing configuration for training.

    :return: loss function, to calculate float
    """
    if config["name"] == "multi_scale":

        def loss(y_true, y_pred):
            return tf.reduce_mean(
                multi_scale_loss(y_true=y_true, y_pred=y_pred, **config["multi_scale"])
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
    Apply the loss at different scales (gaussian smoothing)
    assuming values are between 0 and 1.

    :param y_true: tensor, shape = [batch, dim1, dim2, dim3]
    :param y_pred: tensor, shape = [batch, dim1, dim2, dim3]
    :param loss_type: string, indicating which loss to pass to
                      function single_scale_loss. Supported:
                      [
                          cross-entropy    |
                          mean-squared     |
                          dice             |
                          dice_generalized |
                          jaccard
                      ]
    :param loss_scales: list, values of sigma to pass to func
                        gauss_kernel_1d.
    :return: [batch] of losses.
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
    Calculate the loss on two tensors based on defined
    loss.

    :param y_true: tensor, shape = [batch, dim1, dim2, dim3]
    :param y_pred: tensor, shape = [batch, dim1, dim2, dim3]
    :param loss_type: string, indicating which loss to pass to
                      function single_scale_loss. Supported:
                      [
                          cross-entropy    |
                          mean-squared     |
                          dice             |
                          dice_generalized |
                          jaccard
                      ]
    :return: shape = [batch] of losses
    """
    if loss_type == "cross-entropy":
        return weighted_binary_cross_entropy(y_true, y_pred)
    elif loss_type == "mean-squared":
        return squared_error(y_true, y_pred)
    elif loss_type == "dice":
        return 1 - dice_score(y_true, y_pred)
    elif loss_type == "dice_generalized":
        return 1 - dice_score_generalized(y_true, y_pred)
    elif loss_type == "jaccard":
        return 1 - jaccard_index(y_true, y_pred)
    else:
        raise ValueError("Unknown loss type.")


def squared_error(y_true, y_pred):
    """
    Calculates the mean squared difference between y_true, y_pred.
    - mean((y_true - y_pred)(y_true - y_pred))

    :param y_true: tensor, shape = [batch, dim1, dim2, dim3]
    :param y_pred: tensor, shape = [batch, dim1, dim2, dim3]
    :return: tensor, shape = [batch]
    """
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=[1, 2, 3])


def weighted_binary_cross_entropy(y_true, y_pred, pos_weight=1):
    """
    Calculates weighted binary cross- entropy:
        -loss = − pos_w * y_true log(y_pred) - (1−y_true) log(1−y_pred)
    :param y_true: tensor, shape = [batch, dim1, dim2, dim3]
    :param y_pred: tensor, shape = [batch, dim1, dim2, dim3]
    :param pos_weight: weight of positive class, scalar. Default, [1]
    :return: shape = [batch] of losses.
    """
    y_pred = tf.clip_by_value(y_pred, EPS, 1 - EPS)
    return -pos_weight * tf.reduce_mean(
        y_true * tf.math.log(y_pred), axis=[1, 2, 3]
    ) - tf.reduce_mean((1 - y_true) * tf.math.log(1 - y_pred), axis=[1, 2, 3])


def dice_score(y_true, y_pred, binary=False):
    """
    Calculates dice score:
    - num = 2 * y_true * y_pred
    - denom = y_true + y_pred
    - dice score = num / denom

    where num and denom are summed over the entire image first.
    :param y_true: tensor, shape = [batch, dim1, dim2, dim3]
    :param y_pred: tensor, shape = [batch, dim1, dim2, dim3]
    :param binary: True if the y should be projected to 0 or 1
    :return: shape = [batch] of losses.
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
    Calculates weighted dice score:
    -   let y_prod = y_true * y_pred
        y_sum  = y_true + y_pred
    - num = 2 *  (pos_w * y_true * y_pred + neg_w * (1−y_true) * (1−y_pred))
        = 2 *  ((pos_w+neg_w) * y_prod - neg_w * y_sum + neg_w)
    - denom = (pos_w * (y_true + y_pred) + neg_w * (1−y_true + 1−y_pred))
          = (pos_w-neg_w) * y_sum + 2 * neg_w
    - dice score = num / denom
    Where num and denom are summed over the entire image first.

    :param y_true: tensor, shape = [batch, dim1, dim2, dim3]
    :param y_pred: tensor, shape = [batch, dim1, dim2, dim3]
    :param pos_weight: weight of positive class, default = 1
    :param neg_weight: weight of negative class, default = 0
    :return: shape = [batch] of losses.
    """
    y_prod = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    y_sum = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(
        y_pred, axis=[1, 2, 3]
    )

    numerator = (
        2 * ((pos_weight + neg_weight) * y_prod - neg_weight * y_sum + neg_weight) + EPS
    )
    denominator = (pos_weight - neg_weight) * y_sum + 2 * neg_weight + EPS
    return numerator / denominator


def jaccard_index(y_true, y_pred):
    """
    Calculates jaccard index:
    - num = y_true * y_pred
    - denom = y_true + y_pred - y_true * y_pred
    - jaccard index = num / denom

    :param y_true: tensor, shape = [batch, dim1, dim2, dim3]
    :param y_pred: tensor, shape = [batch, dim1, dim2, dim3]
    :return: shape = [batch] of losses.
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
    """
    Calculates a gaussian kernel.

    :param sigma: number defining standard deviation for
                  gaussian kernel.
    :return: tensor, shape [range(-3*sigma, 3*sigma + 1)]
    """
    if sigma == 0:
        return tf.constant(0)
    else:
        tail = int(sigma * 3)
        k = tf.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)


def cauchy_kernel1d(sigma):
    """
    Approximating cauchy kernel in 1d.

    :param sigma: number, defining standard deviation of kernel.
    :return: tensor, shape [range(-3*sigma, 3*sigma + 1)]
    """
    if sigma == 0:
        return 0
    else:
        tail = int(sigma * 5)
        k = tf.math.reciprocal([((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)


def separable_filter3d(tensor, kernel):
    """
    Creates a 3d separable filter.

    :param tensor: tensor, shape = [batch, dim1, dim2, dim3]
    :param kernel: tensor, shape = [dim4]
    :return: tensor, shape = [batch, dim1, dim2, dim3]
    """
    if len(kernel.shape) == 0:
        return tensor
    else:
        strides = [1, 1, 1, 1, 1]
        tensor = tf.nn.conv3d(
            tf.nn.conv3d(
                tf.nn.conv3d(
                    tf.expand_dims(tensor, axis=4),
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
        return tensor[:, :, :, :, 0]


def compute_centroid(mask, grid):
    """
    Calculate the centroid of the mask.
    :param mask: tensor, shape = [batch, dim1, dim2, dim3]
    :param grid: tensor, shape = [dim1, dim2, dim3, 3]
    :return: shape = [batch, 3], batch of vectors denoting
             location of centroids.
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
    Calculate the L2-distance between two tensors' centroids.

    :param y_true: tensor, shape = [batch, dim1, dim2, dim3]
    :param y_pred: tensor, shape = [batch, dim1, dim2, dim3]
    :param grid: tensor, shape = [dim1, dim2, dim3, 3]
    :return: shape = [batch] of distances.
    """
    centroid_1 = compute_centroid(mask=y_pred, grid=grid)  # shape = [batch, 3]
    centroid_2 = compute_centroid(mask=y_true, grid=grid)  # shape = [batch, 3]
    return tf.sqrt(tf.reduce_sum((centroid_1 - centroid_2) ** 2, axis=[1]))


def foreground_proportion(y):
    """
    Calculating the percentage of foreground vs
    background per 3d volume.

    :param y: tensor, shape = [batch, dim1, dim2, dim3]
    :return: shape = [batch] of foreground percentage.
    """
    y = tf.cast(y >= 0.5, dtype=tf.float32)
    return tf.reduce_sum(y, axis=[1, 2, 3]) / tf.reduce_sum(
        tf.ones_like(y), axis=[1, 2, 3]
    )
