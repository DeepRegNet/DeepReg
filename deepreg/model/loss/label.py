"""
Module provides different loss functions for calculating the dissimilarities between labels.
"""
from typing import Callable

import tensorflow as tf

EPS = tf.keras.backend.epsilon()


def get_dissimilarity_fn(config: dict) -> Callable:
    """
    Parse arguments from a configuration dictionary
    and return the loss by averaging batch loss returned by
    multi- or single-scale loss functions.

    :param config: dict, containing configuration for training.
    :return: loss function, which returns a tensor of shape (batch, )
    """
    if config["name"] == "multi_scale":

        def loss(y_true, y_pred):
            return multi_scale_loss(
                y_true=y_true, y_pred=y_pred, **config["multi_scale"]
            )

        return loss
    elif config["name"] == "single_scale":

        def loss(y_true, y_pred):
            return single_scale_loss(
                y_true=y_true, y_pred=y_pred, **config["single_scale"]
            )

        return loss
    else:
        raise ValueError(f"Unknown loss type {config['name']}.")


def multi_scale_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, loss_type: str, loss_scales: list
) -> tf.Tensor:
    """
    Apply the loss at different scales (gaussian smoothing).
    It is assumed that loss values are between 0 and 1.

    :param y_true: tensor, shape = (batch, dim1, dim2, dim3)
    :param y_pred: tensor, shape = (batch, dim1, dim2, dim3)
    :param loss_type: string, indicating which loss to pass to function single_scale_loss.

      Supported:

      - cross-entropy
      - mean-squared
      - dice
      - dice_generalized
      - jaccard

    :param loss_scales: list, values of sigma to pass to func
                        gauss_kernel_1d.
    :return: (batch,)
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


def single_scale_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, loss_type: str
) -> tf.Tensor:
    """
    Calculate the loss on two tensors based on defined
    loss.

    :param y_true: tensor, shape = (batch, dim1, dim2, dim3)
    :param y_pred: tensor, shape = (batch, dim1, dim2, dim3)
    :param loss_type: string, indicating which loss to pass to
      function single_scale_loss.

      Supported:

      - cross-entropy
      - mean-squared
      - dice
      - dice_generalized
      - jaccard

    :return: shape = (batch,)
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


def squared_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates the mean squared difference between y_true, y_pred.

    mean((y_true - y_pred)(y_true - y_pred))

    :param y_true: tensor, shape = (batch, dim1, dim2, dim3)
    :param y_pred: shape = (batch, dim1, dim2, dim3)
    :return: shape = (batch,)
    """
    return tf.reduce_mean(tf.math.squared_difference(y_true, y_pred), axis=[1, 2, 3])


def weighted_binary_cross_entropy(
    y_true: tf.Tensor, y_pred: tf.Tensor, pos_weight: float = 1
) -> tf.Tensor:
    """
    Calculates weighted binary cross- entropy:

        -loss = − pos_w * y_true log(y_pred) - (1−y_true) log(1−y_pred)

    :param y_true: shape = (batch, dim1, dim2, dim3)
    :param y_pred: shape = (batch, dim1, dim2, dim3)
    :param pos_weight: weight of positive class, scalar. Default value is 1
    :return: shape = (batch,)
    """
    y_pred = tf.clip_by_value(y_pred, 0, 1)
    loss_pos = tf.reduce_mean(y_true * tf.math.log(y_pred + EPS), axis=[1, 2, 3])
    loss_neg = tf.reduce_mean(
        (1 - y_true) * tf.math.log(1 - y_pred + EPS), axis=[1, 2, 3]
    )
    return -pos_weight * loss_pos - loss_neg


def dice_score(y_true: tf.Tensor, y_pred: tf.Tensor, binary: bool = False) -> tf.Tensor:
    """
    Calculates dice score:

    1. num = 2 * y_true * y_pred
    2. denom = y_true + y_pred
    3. dice score = num / denom

    where num and denom are summed over the entire image first.

    :param y_true: shape = (batch, dim1, dim2, dim3)
    :param y_pred: shape = (batch, dim1, dim2, dim3)
    :param binary: True if the y should be projected to 0 or 1
    :return: shape = (batch,)
    """
    if binary:
        y_true = tf.cast(y_true >= 0.5, dtype=tf.float32)
        y_pred = tf.cast(y_pred >= 0.5, dtype=tf.float32)
    numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3]) * 2
    denominator = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(
        y_pred, axis=[1, 2, 3]
    )
    return (numerator + EPS) / (denominator + EPS)


def dice_score_generalized(
    y_true: tf.Tensor, y_pred: tf.Tensor, pos_weight: float = 1, neg_weight: float = 0
) -> tf.Tensor:
    """
    Calculates weighted dice score:

    1. let y_prod = y_true * y_pred and y_sum  = y_true + y_pred
    2. num = 2 *  (pos_w * y_true * y_pred + neg_w * (1−y_true) * (1−y_pred))

       = 2 *  ((pos_w+neg_w) * y_prod - neg_w * y_sum + neg_w)
    3. denom = (pos_w * (y_true + y_pred) + neg_w * (1−y_true + 1−y_pred))

       = (pos_w-neg_w) * y_sum + 2 * neg_w
    4. dice score = num / denom

    where num and denom are summed over the entire image first.

    :param y_true: shape = (batch, dim1, dim2, dim3)
    :param y_pred: shape = (batch, dim1, dim2, dim3)
    :param pos_weight: weight of positive class, default = 1
    :param neg_weight: weight of negative class, default = 0
    :return: shape = (batch,)
    """
    y_prod = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    y_sum = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(
        y_pred, axis=[1, 2, 3]
    )

    numerator = 2 * (
        (pos_weight + neg_weight) * y_prod - neg_weight * y_sum + neg_weight
    )
    denominator = (pos_weight - neg_weight) * y_sum + 2 * neg_weight
    return (numerator + EPS) / (denominator + EPS)


def jaccard_index(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculates jaccard index:

    1. num = y_true * y_pred
    2. denom = y_true + y_pred - y_true * y_pred
    3. jaccard index = num / denom

    :param y_true: shape = (batch, dim1, dim2, dim3)
    :param y_pred: shape = (batch, dim1, dim2, dim3)
    :return: shape = (batch,)
    """
    numerator = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    denominator = (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        - numerator
    )
    return (numerator + EPS) / (denominator + EPS)


def gauss_kernel1d(sigma: int) -> tf.Tensor:
    """
    Calculates a gaussian kernel.

    :param sigma: number defining standard deviation for
                  gaussian kernel.
    :return: shape = (dim, ) or ()
    """
    if sigma == 0:
        return tf.constant(0, tf.float32)
    else:
        tail = int(sigma * 3)
        k = tf.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)


def cauchy_kernel1d(sigma: int) -> tf.Tensor:
    """
    Approximating cauchy kernel in 1d.

    :param sigma: int, defining standard deviation of kernel.
    :return: shape = (dim, ) or ()
    """
    if sigma == 0:
        return tf.constant(0, tf.float32)
    else:
        tail = int(sigma * 5)
        k = tf.math.reciprocal([((x / sigma) ** 2 + 1) for x in range(-tail, tail + 1)])
        return k / tf.reduce_sum(k)


def separable_filter3d(tensor: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    """
    Creates a 3d separable filter.

    Here `tf.nn.conv3d` accepts the `filters` argument of shape
    (filter_depth, filter_height, filter_width, in_channels, out_channels),
    where the first axis of `filters` is the depth not batch,
    and the input to `tf.nn.conv3d` is of shape
    (batch, in_depth, in_height, in_width, in_channels).

    :param tensor: shape = (batch, dim1, dim2, dim3)
    :param kernel: shape = (dim4,)
    :return: shape = (batch, dim1, dim2, dim3)
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


def compute_centroid(mask: tf.Tensor, grid: tf.Tensor) -> tf.Tensor:
    """
    Calculate the centroid of the mask.

    :param mask: shape = (batch, dim1, dim2, dim3)
    :param grid: shape = (dim1, dim2, dim3, 3)
    :return: shape = (batch, 3), batch of vectors denoting
             location of centroids.
    """
    assert len(mask.shape) == 4
    assert len(grid.shape) == 4
    bool_mask = tf.expand_dims(
        tf.cast(mask >= 0.5, dtype=tf.float32), axis=4
    )  # (batch, dim1, dim2, dim3, 1)
    masked_grid = bool_mask * tf.expand_dims(
        grid, axis=0
    )  # (batch, dim1, dim2, dim3, 3)
    numerator = tf.reduce_sum(masked_grid, axis=[1, 2, 3])  # (batch, 3)
    denominator = tf.reduce_sum(bool_mask, axis=[1, 2, 3])  # (batch, 1)
    return (numerator + EPS) / (denominator + EPS)  # (batch, 3)


def compute_centroid_distance(
    y_true: tf.Tensor, y_pred: tf.Tensor, grid: tf.Tensor
) -> tf.Tensor:
    """
    Calculate the L2-distance between two tensors' centroids.

    :param y_true: tensor, shape = (batch, dim1, dim2, dim3)
    :param y_pred: tensor, shape = (batch, dim1, dim2, dim3)
    :param grid: tensor, shape = (dim1, dim2, dim3, 3)
    :return: shape = (batch,)
    """
    centroid_1 = compute_centroid(mask=y_pred, grid=grid)  # (batch, 3)
    centroid_2 = compute_centroid(mask=y_true, grid=grid)  # (batch, 3)
    return tf.sqrt(tf.reduce_sum((centroid_1 - centroid_2) ** 2, axis=1))


def foreground_proportion(y: tf.Tensor) -> tf.Tensor:
    """
    Calculating the percentage of foreground vs
    background per 3d volume.

    :param y: shape = (batch, dim1, dim2, dim3), a 3D label tensor
    :return: shape = (batch,)
    """
    y = tf.cast(y >= 0.5, dtype=tf.float32)
    return tf.reduce_sum(y, axis=[1, 2, 3]) / tf.reduce_sum(
        tf.ones_like(y), axis=[1, 2, 3]
    )
