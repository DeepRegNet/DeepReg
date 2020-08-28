import tensorflow as tf


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
