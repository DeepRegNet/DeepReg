import os

import h5py


def get_h5_sorted_keys(filename):
    with h5py.File(filename, "r") as hf:
        return sorted(hf.keys())


def mkdir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
