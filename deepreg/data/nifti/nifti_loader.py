import nibabel as nib
import numpy as np

from deepreg.data.util import get_sorted_filenames_in_dir


class NiftiFileLoader:
    def __init__(self, dir_path: str, grouped: bool):
        self.dir_path = dir_path
        self.file_paths = get_sorted_filenames_in_dir(dir_path=dir_path, suffix="nii.gz")

    def get_data(self, index: int):
        assert 0 <= index < len(self.file_paths)
        arr = np.asarray(nib.load(self.file_paths[index]).dataobj, dtype=np.float32)
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            arr = arr[:, :, :, 0]
        return arr

    def get_relative_file_paths(self):
        n = len(self.dir_path)
        return [p[n:] for p in self.file_paths]
