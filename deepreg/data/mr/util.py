import h5py


def get_image_shape(filename):
    with h5py.File(filename, "r") as hf:
        keys = sorted([k for k in hf.keys()])
        sh = tuple(hf.get(keys[0]).shape)

        # sanity check
        # all samples have same 4d shape (2, dim1, dim2, dim3)
        assert len(sh) == 4
        for k in keys:
            assert sh == tuple(hf.get(k).shape)
    return sh[1:]


def split_patient_map(pid_map: dict, valid_start, test_start):
    """
    split the map into train / valid / test
    :param pid_map:
    :param valid_start: patient ID, included
    :param test_start: patient ID, included
    :return:
    """
    train, valid, test = dict(), dict(), dict()
    for pid, values in pid_map.items():
        if pid < valid_start:
            train[pid] = values
        elif pid < test_start:
            valid[pid] = values
        else:
            test[pid] = values
    return train, valid, test


def get_patient_map(get_map_func, filename, mode, valid_start, test_start):
    pid_map = get_map_func(filename)
    if mode == "train":
        pid_map, _, _ = split_patient_map(pid_map, valid_start, test_start)
    elif mode == "valid":
        _, pid_map, _ = split_patient_map(pid_map, valid_start, test_start)
    elif mode == "test":
        _, _, pid_map = split_patient_map(pid_map, valid_start, test_start)
    else:
        raise ValueError("Unknown order")
    return pid_map
