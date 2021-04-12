import os
from typing import List, Tuple, Union

import nibabel as nib
import numpy as np

from deepreg.dataset.loader.interface import FileLoader
from deepreg.dataset.util import get_sorted_file_paths_in_dir_with_suffix
from deepreg.registry import REGISTRY

DATA_FILE_SUFFIX = ["nii.gz", "nii"]


def load_nifti_file(file_path: str) -> np.ndarray:
    """
    :param file_path: path of a Nifti file with suffix .nii or .nii.gz
    :return: return the numpy array
    """
    if not (file_path.endswith(".nii") or file_path.endswith(".nii.gz")):
        raise ValueError(
            f"Nifti file path must end with .nii or .nii.gz, got {file_path}."
        )
    return np.asarray(nib.load(file_path).dataobj, dtype=np.float32)


@REGISTRY.register_file_loader(name="nifti")
class NiftiFileLoader(FileLoader):
    """Generalized loader for nifti files."""

    def __init__(self, dir_paths: List[str], name: str, grouped: bool):
        """
        Init.

        :param dir_paths: path of directories having nifti files.
        :param name: name is used to identify the subdirectories.
        :param grouped: whether the data is grouped.
        """
        super().__init__(dir_paths=dir_paths, name=name, grouped=grouped)
        self.data_path_splits = None
        self.set_data_structure()
        self.group_struct = None
        if self.grouped:
            self.set_group_structure()

    def set_data_structure(self):
        """
        Store the data structure in the memory so that
        we can retrieve data using data_index
        this function sets data_path_splits,
        a list of string tuples to identify path of data

        - if grouped, a split is (dir_path, group_path, file_name, suffix)
          data is stored in dir_path/name/group_path/file_name.suffix

        - if not grouped, a split is (dir_path, file_name, suffix)
          data is stored in dir_path/name/file_name.suffix

        """
        data_path_splits = []
        for dir_path in self.dir_paths:
            # dir_path/name
            named_dir_path = os.path.join(dir_path, self.name)
            assert os.path.exists(
                named_dir_path
            ), f"directory {named_dir_path} does not exist"
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
        if len(data_path_splits) == 0:
            raise ValueError(
                f"No data collected from {self.dir_paths} in NiftiFileLoader, "
                f"please verify the path is correct."
            )
        self.data_path_splits = sorted(data_path_splits)

    def set_group_structure(self):
        """
        In addition to set_data_structure
        store the group structure in the group_struct so that
        group_struct[group_index] = list of data_index
        we can retrieve data using (group_index, in_group_data_index)
        data_index = group_struct[group_index][in_group_data_index]
        """
        # group_struct_dict[group_id] = list of data_index
        group_struct_dict = {}
        for data_index, split in enumerate(self.data_path_splits):
            # we use (dir_path, group_path) as group_id
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
        if isinstance(index, int):  # paired or unpaired
            assert not self.grouped
            assert 0 <= index
            data_index = index
        elif isinstance(index, tuple):  # grouped
            assert self.grouped
            group_index, in_group_data_index = index
            assert 0 <= group_index
            assert 0 <= in_group_data_index
            data_index = self.group_struct[group_index][in_group_data_index]  # type: ignore
        else:
            raise ValueError(
                f"index for NiftiFileLoader.get_data must be int, "
                f"or tuple of length two, got {index}"
            )
        # if not grouped:
        #   path  = dir_path/name/file_name.suffix
        #   split = (dir_path, file_name, suffix)
        # else:
        #   path  = dir_path/name/group_path/file_name.suffix
        #   split = (dir_path, group_path, file_name, suffix)
        path_splits = self.data_path_splits[data_index]  # type: ignore
        path_splits, suffix = path_splits[:-1], path_splits[-1]
        path_splits = path_splits[:1] + (self.name,) + path_splits[1:]
        file_path = os.path.join(*path_splits) + "." + suffix

        arr = load_nifti_file(file_path=file_path)
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            # currently have not encountered
            arr = arr[:, :, :, 0]  # pragma: no cover
        return arr

    def get_data_ids(self) -> List:
        """
        Return the unique IDs of the data in this data set
        this function is used to verify the consistency between
        images and label, moving and fixed.

        :return: data_path_splits but without suffix
        """
        return [x[:-1] for x in self.data_path_splits]  # type: ignore

    def get_num_images(self) -> int:
        """
        :return: int, number of images in this data set
        """
        return len(self.data_path_splits)  # type: ignore

    def close(self):
        """Close opened files."""
        return
