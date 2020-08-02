import os

import nibabel as nib
import numpy as np

from deepreg.dataset.loader.interface import FileLoader
from deepreg.dataset.util import get_sorted_filenames_in_dir


def load_nifti_file(filepath):
    if not (filepath.endswith(".nii") or filepath.endswith(".nii.gz")):
        raise ValueError(
            f"Nifti file path must end with .nii or .nii.gz, got {filepath}."
        )
    return np.asarray(nib.load(filepath).dataobj, dtype=np.float32)


class NiftiFileLoader(FileLoader):
    """Generalized loader for nifti files"""

    def __init__(self, dir_path: str, name: str, grouped: bool):
        super(NiftiFileLoader, self).__init__(
            dir_path=dir_path, name=name, grouped=grouped
        )
        self.file_paths = get_sorted_filenames_in_dir(
            dir_path=os.path.join(dir_path, name), suffix="nii.gz"
        )
        self.set_group_structure()

    def get_data(self, index: (int, tuple)):
        if isinstance(index, int):  # paired or unpaired
            assert not self.grouped
            assert 0 <= index < len(self.file_paths)
            filepath = self.file_paths[index]
        elif isinstance(index, tuple):  # grouped
            assert self.grouped
            if len(index) == 2:
                group_index, sample_index = index
                assert 0 <= group_index < len(self.group_ids)
                assert 0 <= sample_index
                assert sample_index < len(
                    self.group_sample_dict[self.group_ids[group_index]]
                )
                filepath = self.group_sample_dict[self.group_ids[group_index]][
                    sample_index
                ]
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
        n = len(os.path.join(self.dir_path, self.name))
        return [p[n:] for p in self.file_paths]

    def get_num_images(self) -> int:
        return len(self.file_paths)

    def set_group_structure(self):
        if self.grouped:
            group_sample_dict = dict()  # dict[group_path] = [file_path]
            for path in self.file_paths:
                head, _ = os.path.split(path)
                if head not in group_sample_dict.keys():
                    group_sample_dict[head] = []
                group_sample_dict[head].append(path)
            for k in group_sample_dict.keys():  # sort paths under the same group
                group_sample_dict[k] = sorted(group_sample_dict[k])
            self.group_ids = sorted(list(group_sample_dict.keys()))
            self.group_sample_dict = group_sample_dict

    def close(self):
        pass
