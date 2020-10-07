from typing import List

import numpy as np
import tensorflow as tf


def is_equal_np(x: (np.ndarray, List), y: (np.ndarray, List)) -> bool:
    """return true if two numpy arrays are nearly equal"""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return np.all(np.isclose(x, y))


def is_equal_tf(
    x: (tf.Tensor, np.ndarray, List),
    y: (tf.Tensor, np.ndarray, List),
    atol: float = 1.0e-8,
) -> bool:
    """return true if two tf tensors are nearly equal"""
    x = tf.cast(x, dtype=tf.float32).numpy()
    y = tf.cast(y, dtype=tf.float32).numpy()
    return np.all(np.isclose(x, y, atol=atol))
