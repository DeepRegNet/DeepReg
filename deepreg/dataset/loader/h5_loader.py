"""
Load h5 files and associated information.
"""
import os
from typing import List, Tuple, Union

import h5py
import numpy as np

from deepreg.dataset.loader.interface import FileLoader
from deepreg.registry import REGISTRY

DATA_KEY_FORMAT = "group-{}-{}"


@REGISTRY.register_file_loader(name="h5")
class H5FileLoader(FileLoader):
    """Generalized loader for h5 files."""

    def __init__(self, dir_paths: List[str], name: str, grouped: bool):
        """
        Init.

        :param dir_paths: path of h5 files.
        :param name: name is used to identify the file names.
        :param grouped: whether the data is grouped.
        """
        super().__init__(dir_paths=dir_paths, name=name, grouped=grouped)
        self.h5_files = None
        self.data_path_splits = None
        self.set_data_structure()
        self.group_struct = None
        if self.grouped:
            self.set_group_structure()

    def set_data_structure(self):
        """
        Store the data structure in  memory so that
        we can retrieve data using data_index.
        This function sets two attributes:

        - h5_files, a dict such that h5_files[dir_path] = opened h5 file handle
        - data_path_splits, a list of string tuples to identify path of data

          - if grouped, a split is (dir_path, group_name, data_key) such that
            data = h5_files[dir_path]["group-{group_name}-{data_key}"]
          - if not grouped, a split is (dir_path, data_key) such that
            data = h5_files[dir_path][data_key]
        """
        h5_files = {}
        data_path_splits = []
        for dir_path in self.dir_paths:
            h5_file_path = os.path.join(dir_path, self.name + ".h5")
            assert os.path.exists(
                h5_file_path
            ), f"h5 file {h5_file_path} does not exist"
            h5_file = h5py.File(h5_file_path, "r")
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
        if len(data_path_splits) == 0:
            raise ValueError(
                f"No data collected from {self.dir_paths} in H5FileLoader, "
                f"please verify the path is correct."
            )
        self.h5_files = h5_files
        self.data_path_splits = data_path_splits

    def set_group_structure(self):
        """
        Similar to NiftiLoader
        as the first two tokens of a split forms a group_id.
        Store the group structure in group_struct so that
        group_struct[group_index] = list of data_index.
        Retrieve data using (group_index, in_group_data_index).
        data_index = group_struct[group_index][in_group_data_index].
        """
        # group_struct_dict[group_id] = list of data_index
        group_struct_dict = {}
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

    def get_data(self, index: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Get one data array by specifying an index

        :param index: the data index which is required

          - for paired or unpaired, the index is one single int, data_index
          - for grouped, the index is a tuple of two ints,
            (group_index, in_group_data_index)
        :returns arr: the data array at the specified index
        """
        assert self.data_path_splits is not None
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
                f"index for H5FileLoader.get_data must be int, "
                f"or tuple of length two, got {index}"
            )
        arr = np.asarray(self.h5_files[dir_path][data_key], dtype=np.float32)
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            # currently have not encountered
            arr = arr[:, :, :, 0]  # pragma: no cover
        return arr

    def get_data_ids(self) -> List:
        """
        Get the unique IDs of data in this data set to
        verify consistency between
        images and label, moving and fixed.

        :return: data_path_splits as the data can be identified
            using dir_path and data_key
        """
        return self.data_path_splits  # type: ignore

    def get_num_images(self) -> int:
        """
        :return: int, number of images in this data set
        """
        return len(self.data_path_splits)  # type: ignore

    def close(self):
        """Close opened h5 file handles."""
        for f in self.h5_files.values():
            f.close()
