import nibabel as nib
import numpy as np

from deepreg.data.util import get_sorted_filenames_in_dir


class NiftiFileLoader:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.file_paths = get_sorted_filenames_in_dir(dir_path=dir_path, suffix="nii.gz")

    def get_data(self, index: int):
        assert 0 <= index < len(self.file_paths)
        return np.asarray(nib.load(self.file_paths[index]).dataobj, dtype=np.float32)

    def get_relative_file_paths(self):
        n = len(self.dir_path)
        return [p[n:] for p in self.file_paths]
