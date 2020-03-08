import tensorflow as tf

EPS = 1.0e-6  # epsilon to prevent NaN


def similarity_fn(y_true, y_pred, name, **kwargs):
    """

    :param y_true: fixed_image, shape = [batch, f_dim1, f_dim2, f_dim3]
    :param y_pred: warped_moving_image, shape = [batch, f_dim1, f_dim2, f_dim3]
    :param name:
    :return:
    """
    y_true = tf.expand_dims(y_true, axis=4)
    y_pred = tf.expand_dims(y_pred, axis=4)
    if name == "lncc":
        return -local_normalized_cross_correlation(y_true, y_pred)
    else:
        raise ValueError("Unknown loss type.")


def local_normalized_cross_correlation(y_true, y_pred, kernel_size=9):
    """
    moving a kernel/window on the y_true/y_pred
    then calculate the ncc in the window of y_true/y_pred
    average over all windows in the end

    :param y_true: shape = [batch, dim1, dim2, dim3, ch]
    :param y_pred: shape = [batch, dim1, dim2, dim3, ch]
    :param kernel_size:
    :return: shape = [batch]
    """

    kernel_vol = kernel_size ** 3
    filters = tf.ones(shape=[kernel_size, kernel_size, kernel_size, 1, 1])  # [dim1, dim2, dim3, d_in, d_out]
    strides = [1, 1, 1, 1, 1]
    padding = "SAME"

    # t = y_true, p = y_pred
    t2 = y_true * y_true
    p2 = y_pred * y_pred
    tp = y_true * y_pred

    t_sum = tf.nn.conv3d(y_true, filters=filters, strides=strides, padding=padding)
    p_sum = tf.nn.conv3d(y_pred, filters=filters, strides=strides, padding=padding)
    t2_sum = tf.nn.conv3d(t2, filters=filters, strides=strides, padding=padding)
    p2_sum = tf.nn.conv3d(p2, filters=filters, strides=strides, padding=padding)
    tp_sum = tf.nn.conv3d(tp, filters=filters, strides=strides, padding=padding)

    t_avg = t_sum / kernel_vol
    p_avg = p_sum / kernel_vol

    cross = tp_sum - p_avg * t_sum
    t_var = t2_sum - t_avg * t_sum
    p_var = p2_sum - p_avg * p_sum

    ncc = (cross * cross + EPS) / (t_var * p_var + EPS)
    return tf.reduce_mean(ncc, axis=[1, 2, 3, 4])
