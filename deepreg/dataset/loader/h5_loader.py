"""
Loads h5 files and some associated information
"""
import os

import h5py
import numpy as np

from deepreg.dataset.loader.interface import FileLoader

DATA_KEY_FORMAT = "group-{}-{}"


class H5FileLoader(FileLoader):
    """Generalized loader for h5 files"""

    def __init__(self, dir_path: str, name: str, grouped: bool):
        super(H5FileLoader, self).__init__(
            dir_path=dir_path, name=name, grouped=grouped
        )
        self.h5_file = h5py.File(os.path.join(dir_path, name + ".h5"), "r")
        self.data_keys = sorted(self.h5_file.keys())
        self.set_group_structure()

    def get_data(self, index: (int, tuple)):
        """
        Get one data array by specifying an index

        :param index: the data index which is required

        :returns arr: the data array at the specified index
        """
        if isinstance(index, int):  # paired or unpaired
            assert not self.grouped
            assert 0 <= index < len(self.data_keys)
            data_key = self.data_keys[index]
        elif isinstance(index, tuple):
            assert self.grouped
            group_index, sample_index = index
            group_id = self.group_ids[group_index]
            sample_id = self.group_sample_dict[group_id][sample_index]
            data_key = DATA_KEY_FORMAT.format(group_id, sample_id)
        else:
            raise ValueError(
                "index for H5FileLoader.get_data must be int, or tuple of length two, got {}".format(
                    index
                )
            )
        arr = np.asarray(self.h5_file[data_key], dtype=np.float32)
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            arr = arr[:, :, :, 0]
        return arr

    def get_data_ids(self):
        return self.data_keys

    def get_num_images(self) -> int:
        return len(self.data_keys)

    def set_group_structure(self):
        if self.grouped:
            group_sample_dict = dict()
            for data_key in self.data_keys:
                tokens = data_key.split("-")
                group_id, sample_id = tokens[-2], tokens[-1]
                if group_id not in group_sample_dict.keys():
                    group_sample_dict[group_id] = []
                group_sample_dict[group_id].append(sample_id)
            for group_id in group_sample_dict.keys():
                group_sample_dict[group_id] = sorted(group_sample_dict[group_id])
            self.group_ids = sorted(list(group_sample_dict.keys()))
            self.group_sample_dict = group_sample_dict

    def close(self):
        self.h5_file.close()
