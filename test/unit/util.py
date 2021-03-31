from typing import List, Union

import numpy as np
import tensorflow as tf


def is_equal_np(
    x: Union[np.ndarray, List], y: Union[np.ndarray, List], atol: float = 1.0e-7
) -> bool:
    """
    Check if two numpy arrays are identical.

    :param x:
    :param y:
    :param atol: error margin
    :return: return true if two tf tensors are nearly equal
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return x.shape == y.shape and np.all(np.isclose(x, y, atol=atol))


def is_equal_tf(
    x: Union[tf.Tensor, np.ndarray, List],
    y: Union[tf.Tensor, np.ndarray, List],
    atol: float = 1.0e-7,
) -> bool:
    """
    Check if two tf tensors are identical.

    :param x:
    :param y:
    :param atol: error margin
    :return: return true if two tf tensors are nearly equal
    """
    x = tf.cast(x, dtype=tf.float32).numpy()
    y = tf.cast(y, dtype=tf.float32).numpy()
    return x.shape == y.shape and np.all(np.isclose(x, y, atol=atol))
