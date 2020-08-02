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
