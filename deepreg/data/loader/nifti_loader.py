import os

import nibabel as nib
import numpy as np

from deepreg.data.loader.interface import FileLoader
from deepreg.data.util import get_sorted_filenames_in_dir


class NiftiFileLoader(FileLoader):
    """Generalized loader for nifti files"""

    def __init__(self, dir_path: str, grouped: bool):
        super(NiftiFileLoader, self).__init__(dir_path=dir_path, grouped=grouped)
        self.file_paths = get_sorted_filenames_in_dir(dir_path=dir_path, suffix="nii.gz")
        if grouped:
            file_path_dict = dict()  # dict[group_path] = [file_path]
            for path in self.file_paths:
                head, _ = os.path.split(path)
                if head not in file_path_dict.keys():
                    file_path_dict[head] = []
                file_path_dict[head].append(path)
            for k in file_path_dict.keys():  # sort paths under the same group
                file_path_dict[k] = sorted(file_path_dict[k])
            self.group_paths = sorted(list(file_path_dict.keys()))
            self.file_path_dict = file_path_dict

    def get_data(self, index: (int, tuple)):
        filepath = None
        if isinstance(index, int):
            assert not self.grouped
            assert 0 <= index < len(self.file_paths)
            filepath = self.file_paths[index]
        elif isinstance(index, tuple):
            assert self.grouped
            if len(index) == 2:
                group_index, sample_index = index
                assert 0 <= group_index < len(self.group_paths)
                assert 0 <= sample_index < len(self.file_path_dict[self.group_paths[group_index]])
                filepath = self.file_path_dict[self.group_paths[group_index]][sample_index]
        if filepath is None:
            raise ValueError("Index for get_data of NiftiFileLoader must be int or tuple of length two")

        arr = np.asarray(nib.load(filepath).dataobj, dtype=np.float32)
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            arr = arr[:, :, :, 0]
        return arr

    def get_num_images(self) -> int:
        return len(self.file_paths)

    def get_data_ids(self):
        n = len(self.dir_path)
        return [p[n:] for p in self.file_paths]

    def get_num_images_per_group(self):
        assert self.grouped
        return [len(self.file_path_dict[g])
                for g in self.group_paths]

    def get_num_groups(self) -> int:
        assert self.grouped
        return len(self.group_paths)
