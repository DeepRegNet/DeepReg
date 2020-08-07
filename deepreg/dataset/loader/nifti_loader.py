import os
from typing import List

import nibabel as nib
import numpy as np

from deepreg.dataset.loader.interface import FileLoader
from deepreg.dataset.util import get_sorted_file_paths_in_dir_with_suffix

DATA_FILE_SUFFIX = ["nii.gz", "nii"]


def load_nifti_file(filepath):
    if not (filepath.endswith(".nii") or filepath.endswith(".nii.gz")):
        raise ValueError(
            f"Nifti file path must end with .nii or .nii.gz, got {filepath}."
        )
    return np.asarray(nib.load(filepath).dataobj, dtype=np.float32)


class NiftiFileLoader(FileLoader):
    """Generalized loader for nifti files"""

    def __init__(self, dir_paths: List[str], name: str, grouped: bool):
        super(NiftiFileLoader, self).__init__(
            dir_paths=dir_paths, name=name, grouped=grouped
        )
        self.data_path_splits = None
        self.set_data_structure()
        if self.grouped:
            self.group_struct = None
            self.set_group_structure()

    def set_data_structure(self):
        data_path_splits = []
        for dir_path in self.dir_paths:
            # if not grouped:
            #   data_path is dir_path/name/file_name.suffix
            #   here, file_name can have nested directories
            # else:
            #   data_path is dir_path/name/group_path/file_name.suffix

            # dir_path/name
            named_dir_path = os.path.join(dir_path, self.name)
            # each element is (file_path, suffix)
            data_paths = get_sorted_file_paths_in_dir_with_suffix(
                dir_path=named_dir_path, suffix=DATA_FILE_SUFFIX
            )
            if self.grouped:
                # each element is (dir_path, group_path, file_name, suffix)
                data_path_splits += [
                    (dir_path,) + os.path.split(file_path) + (suffix,)
                    for file_path, suffix in data_paths
                ]
            else:
                # each element is (dir_path, file_name, suffix)
                data_path_splits += [
                    (dir_path, file_path, suffix) for file_path, suffix in data_paths
                ]
        self.data_path_splits = sorted(data_path_splits)

    def set_group_structure(self):
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
            data_index = index
        elif isinstance(index, tuple):  # grouped
            assert self.grouped
            group_index, in_group_data_index = index
            assert 0 <= group_index
            assert 0 <= in_group_data_index
            data_index = self.group_struct[group_index][in_group_data_index]
        else:
            raise ValueError(
                "index for NiftiFileLoader.get_data must be int, or tuple of length two, got {}".format(
                    index
                )
            )
        # if not grouped:
        #   path  = dir_path/name/file_name.suffix
        #   split = (dir_path, file_name, suffix)
        # else:
        #   path  = dir_path/name/group_path/file_name.suffix
        #   split = (dir_path, group_path, file_name, suffix)
        path_splits = self.data_path_splits[data_index]
        path_splits, suffix = path_splits[:-1], path_splits[-1]
        path_splits = path_splits[:1] + (self.name,) + path_splits[1:]
        file_path = os.path.join(*path_splits) + "." + suffix

        arr = load_nifti_file(filepath=file_path)
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            arr = arr[:, :, :, 0]
        return arr

    def get_data_ids(self):
        """data_id does not need suffix"""
        return [x[:-1] for x in self.data_path_splits]

    def get_num_images(self) -> int:
        return len(self.data_path_splits)

    def close(self):
        pass
