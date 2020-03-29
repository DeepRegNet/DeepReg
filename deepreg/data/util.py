import h5py


def get_h5_sorted_keys(filename):
    with h5py.File(filename, "r") as hf:
        return sorted(hf.keys())
