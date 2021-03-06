from typing import List, Union

import numpy as np


def normalize_array(arr: np.ndarray, v_min=None, v_max=None) -> np.ndarray:
    """
    Normalize a numpy array.

    The array is normalized such that
    its values are normalized from [v_min, v_max] to [0, 1].
    If min/max are not provided, will use the min/max of the array.
    Values outside of [v_min, v_max] will be clipped.

    :param arr: array to be normalized
    :param v_min: minimum of the value before normalization.
    :param v_max: maximum of the value before normalization.
    :return: normalized array.
    """
    v_min = np.min(arr) if v_min is None else v_min
    v_max = np.max(arr) if v_max is None else v_max
    if v_min == v_max:
        return arr * 0
    assert v_min < v_max
    arr = np.clip(a=arr, a_min=v_min, a_max=v_max)
    arr = (arr - v_min) / (v_max - v_min)
    return arr


def remove_prefix_suffix(
    x: str, prefix: Union[str, List[str]], suffix: Union[str, List[str]]
) -> str:
    """
    Remove the prefix and suffix from a string,
    prefix and suffix can be a string or a list of strings.
    :param x: input string
    :param prefix: a string or a list of strings, will check prefix one by one
    :param suffix: a string or a list of strings, will check suffix one by one
    :return: the string without prefix/suffix
    """
    if isinstance(prefix, str):
        prefix = [prefix]
    if isinstance(suffix, str):
        suffix = [suffix]

    for s in prefix:
        if x.startswith(s):
            x = x[len(s) :]
            break

    for s in suffix:
        if x.endswith(s):
            x = x[: -len(s)]
            break

    return x
