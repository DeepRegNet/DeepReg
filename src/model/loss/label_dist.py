import tensorflow as tf

EPS = 1.e0 - 6  # epsilon to prevent NaN


def compute_centroid(mask, grid):
    """
    calculate the centroid of the mask
    :param mask: shape = [batch, dim1, dim2, dim3]
    :param grid: shape = [dim1, dim2, dim3, 3]
    :return: shape = [batch, 3]
    """
    bool_mask = tf.expand_dims(tf.cast(mask >= 0.5, dtype=tf.float32), axis=4)  # [batch, dim1, dim2, dim3, 1]
    masked_grid = bool_mask * tf.expand_dims(grid, axis=0)  # [batch, dim1, dim2, dim3, 3]
    numerator = tf.reduce_sum(masked_grid, axis=[1, 2, 3]) + EPS  # [batch, 3]
    denominator = tf.reduce_sum(bool_mask, axis=[1, 2, 3]) + EPS  # [batch, 1]
    return numerator / denominator  # [batch, 3]


def compute_centroid_distance(y_true, y_pred, grid):
    """
    :param y_true: shape = [batch, dim1, dim2, dim3]
    :param y_pred: shape = [batch, dim1, dim2, dim3]
    :param grid: shape = [dim1, dim2, dim3, 3]
    :return:
    """
    c1 = compute_centroid(mask=y_pred, grid=grid)
    c2 = compute_centroid(mask=y_true, grid=grid)
    return tf.sqrt(tf.reduce_sum((c1 - c2) ** 2))
