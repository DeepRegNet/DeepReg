import numpy as np
import tensorflow as tf

EPS = 1e-6


def is_equal_np(x: (np.ndarray, list, tuple), y: (np.ndarray, list, tuple)) -> bool:
    """return true if two numpy arrays are nearly equal"""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return np.max(np.abs(x - y)) < EPS


def is_equal_tf(x: tf.Tensor, y: tf.Tensor) -> bool:
    """return true if two tf tensors are nearly equal"""
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.float32)
    return tf.reduce_max(tf.abs(x - y)).numpy() < EPS
