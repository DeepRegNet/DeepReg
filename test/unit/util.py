from typing import List, Union

import numpy as np
import tensorflow as tf

from deepreg.constant import EPS


def is_equal_np(
    x: Union[np.ndarray, List], y: Union[np.ndarray, List], atol: float = EPS
) -> bool:
    """
    Check if two numpy arrays are identical within a tolerance.

    :param x:
    :param y:
    :param atol: error margin
    :return: return true if two tf tensors are nearly equal
    """
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    # check shape
    if x.shape != y.shape:
        return False

    # check nan values
    # support case some values are nan
    if np.any(np.isnan(x) != np.isnan(y)):
        return False
    x = np.nan_to_num(x)
    y = np.nan_to_num(y)

    # check values
    return np.all(np.isclose(x, y, atol=atol))


def is_equal_tf(
    x: Union[tf.Tensor, np.ndarray, List],
    y: Union[tf.Tensor, np.ndarray, List],
    atol: float = EPS,
) -> bool:
    """
    Check if two tf tensors are identical within a tolerance.

    :param x:
    :param y:
    :param atol: error margin
    :return: return true if two tf tensors are nearly equal
    """
    x = tf.cast(x, dtype=tf.float32).numpy()
    y = tf.cast(y, dtype=tf.float32).numpy()
    return is_equal_np(x=x, y=y, atol=atol)
