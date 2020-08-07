import os
from typing import List

import nibabel as nib
import numpy as np

from deepreg.dataset.loader.interface import FileLoader
from deepreg.dataset.loader.util import remove_prefix_suffix
from deepreg.dataset.util import get_sorted_file_paths_in_dir_with_suffix


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
        self.file_paths = [
            get_sorted_file_paths_in_dir_with_suffix(
                dir_paths=[os.path.join(p, name)], suffix=["nii.gz", "nii"]
            )
            for p in dir_paths
        ]
        self.data_keys = self.get_data_keys(file_paths=self.file_paths)
        self.set_group_structure()

    @staticmethod
    def get_data_keys(file_paths):
        """
        each data key is (dir_index, path_index) such that
        data_path = self.file_paths[dir_index][path_index]
        """
        data_keys = []
        for i, paths in enumerate(file_paths):
            data_keys += [(i, j) for j, _ in enumerate(paths)]
        data_keys = sorted(data_keys)
        return data_keys

    def get_data(self, index: (int, tuple)):
        if isinstance(index, int):  # paired or unpaired
            assert not self.grouped
            assert 0 <= index < len(self.data_keys)
            dir_index, path_index = self.data_keys[index]
            filepath = self.file_paths[dir_index][path_index]
        elif isinstance(index, tuple):  # grouped
            assert self.grouped
            group_index, sample_index = index
            assert 0 <= group_index < len(self.group_ids)
            assert 0 <= sample_index
            assert sample_index < len(
                self.group_sample_dict[self.group_ids[group_index]]
            )
            filepath = self.group_sample_dict[self.group_ids[group_index]][sample_index]
        else:
            raise ValueError(
                "index for NiftiFileLoader.get_data must be int, or tuple of length two, got {}".format(
                    index
                )
            )

        arr = load_nifti_file(filepath=filepath)
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            arr = arr[:, :, :, 0]
        return arr

    def get_data_ids(self):
        data_ids = []
        for dir_index, dir_path in enumerate(self.dir_paths):
            prefix = os.path.join(dir_path, self.name)
            suffix = [".nii.gz", ".nii"]
            data_ids += [
                remove_prefix_suffix(x=p, prefix=prefix, suffix=suffix)
                for p in self.file_paths[dir_index]
            ]
        return data_ids

    def get_num_images(self) -> int:
        return len(self.data_keys)

    def set_group_structure(self):
        if self.grouped:
            group_sample_dict = dict()  # dict[(file_index, group_path)] = [file_path]
            for dir_index, paths in enumerate(self.file_paths):
                # paths are nifti file paths under the dir_index-th directory
                for path in paths:
                    head, _ = os.path.split(path)
                    group_id = (dir_index, head)
                    if group_id not in group_sample_dict.keys():
                        group_sample_dict[group_id] = []
                    group_sample_dict[group_id].append(path)
            for k in group_sample_dict.keys():  # sort paths under the same group
                group_sample_dict[k] = sorted(group_sample_dict[k])
            self.group_ids = sorted(list(group_sample_dict.keys()))
            self.group_sample_dict = group_sample_dict

    def close(self):
        pass
