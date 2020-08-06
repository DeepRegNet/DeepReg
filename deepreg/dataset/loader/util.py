import numpy as np

EPS = 1.0e-6


def normalize_array(arr: np.ndarray, v_min=None, v_max=None) -> np.ndarray:
    """
    Normalize a numpy array such that
    its values are normalized from [v_min, v_max] to [0, 1]
    If min/max are not provided, will use the min/max of the array
    Values outside of [v_min, v_max] will be clipped
    """
    v_min = np.min(arr) if v_min is None else v_min
    v_max = np.max(arr) if v_max is None else v_max
    assert v_min <= v_max
    arr = np.clip(a=arr, a_min=v_min, a_max=v_max)
    arr = (arr - v_min + EPS) / (v_max - v_min + EPS)
    return arr


def remove_prefix_suffix(x: str, prefix: (str, list), suffix: (str, list)) -> str:
    """
    remove the prefix and suffix from a string,
    prefix and suffix can be a string or a list of strings
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
