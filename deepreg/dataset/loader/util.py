import numpy as np

EPS = 1.0e-6


def normalize_array(arr: np.ndarray) -> np.ndarray:
    min, max = np.min(arr), np.max(arr)
    arr = (arr - min) / (max - min + EPS)
    return arr
