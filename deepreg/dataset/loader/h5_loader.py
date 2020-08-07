"""
Loads h5 files and some associated information
"""
import os
from typing import List

import h5py
import numpy as np

from deepreg.dataset.loader.interface import FileLoader

DATA_KEY_FORMAT = "group-{}-{}"


class H5FileLoader(FileLoader):
    """Generalized loader for h5 files"""

    def __init__(self, dir_paths: List[str], name: str, grouped: bool):
        super(H5FileLoader, self).__init__(
            dir_paths=dir_paths, name=name, grouped=grouped
        )
        self.h5_files = None
        self.data_path_splits = None
        self.set_data_structure()
        if self.grouped:
            self.group_struct = None
            self.set_group_structure()

    def set_data_structure(self):
        h5_files = dict()
        data_path_splits = []
        for dir_path in self.dir_paths:
            h5_file = h5py.File(os.path.join(dir_path, self.name + ".h5"), "r")
            h5_files[dir_path] = h5_file

            if self.grouped:
                # each element is (dir_path, group_name, data_key)
                # check h5 file keys
                key_splits = [k.split("-") for k in sorted(h5_file.keys())]
                assert all(
                    [len(x) == 3 and x[0] == "group" for x in key_splits]
                ), f"h5_file keys must be of form group-X-Y, got {key_splits}"
                data_path_splits += [(dir_path, k[1], k[2]) for k in key_splits]
            else:
                # each element is (dir_path, data_key)
                data_path_splits += [(dir_path, k) for k in sorted(h5_file.keys())]
        self.h5_files = h5_files
        self.data_path_splits = data_path_splits

    def set_group_structure(self):
        """same code as NiftiLoader"""
        # group_struct_dict[group_id] = list of data_index
        group_struct_dict = dict()
        for data_index, split in enumerate(self.data_path_splits):
            group_id = split[:2]
            if group_id not in group_struct_dict.keys():
                group_struct_dict[group_id] = []
            group_struct_dict[group_id].append(data_index)
        # group_struct[group_index] = list of data_index
        group_struct = []
        for k in sorted(group_struct_dict.keys()):
            group_struct.append(group_struct_dict[k])
        self.group_struct = group_struct

    def get_data(self, index: (int, tuple)) -> np.ndarray:
        """
        Get one data array by specifying an index
        :param index: the data index which is required
        :returns arr: the data array at the specified index
        """
        if isinstance(index, int):  # paired or unpaired
            assert not self.grouped
            assert 0 <= index
            dir_path, data_key = self.data_path_splits[index]
        elif isinstance(index, tuple):
            assert self.grouped
            group_index, in_group_data_index = index
            assert 0 <= group_index
            assert 0 <= in_group_data_index
            data_index = self.group_struct[group_index][in_group_data_index]
            dir_path, group_name, data_key = self.data_path_splits[data_index]
            data_key = DATA_KEY_FORMAT.format(group_name, data_key)
        else:
            raise ValueError(
                "index for H5FileLoader.get_data must be int, or tuple of length two, got {}".format(
                    index
                )
            )
        arr = np.asarray(self.h5_files[dir_path][data_key], dtype=np.float32)
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            arr = arr[:, :, :, 0]
        return arr

    def get_data_ids(self):
        return self.data_path_splits

    def get_num_images(self) -> int:
        return len(self.data_path_splits)

    def close(self):
        for f in self.h5_files.values():
            f.close()
