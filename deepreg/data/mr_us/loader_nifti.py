import os

import nibabel as nib
import numpy as np

from deepreg.data.loader_basic import BasicDataLoader
from deepreg.data.mr_us.util import get_label_indices


class NiftiFileLoader:
    def __init__(self, dir_name, load_into_memory):
        file_names, file_path_names = get_fnames_in_dir(dir_name)
        self.load_into_memory = load_into_memory
        self.loaded_data = None
        self.name_to_path_dict = None
        if self.load_into_memory:
            self.loaded_data = dict(zip(file_names,
                                        [load_from_nifti(x) for x in file_path_names]))

        else:
            self.name_to_path_dict = dict(zip(file_names, file_path_names))

    def get_data(self, file_name):
        if self.load_into_memory:
            return self.loaded_data[file_name]
        else:
            return load_from_nifti(self.name_to_path_dict[file_name])

    def get_sorted_keys(self):
        if self.load_into_memory:
            return sorted(self.loaded_data.keys())
        else:
            return sorted(self.name_to_path_dict.keys())

    def get_image_shape(self):
        keys = self.get_sorted_keys()
        data = self.get_data(keys[0])
        return data.shape[:3]  # label.shape might be [dim1, dim2, dim3, num_labels]


class NiftiDataLoader(BasicDataLoader):
    def __init__(self,
                 moving_image_dir, fixed_image_dir, moving_label_dir, fixed_label_dir,
                 load_into_memory, sample_label):
        super(NiftiDataLoader, self).__init__()
        loader_moving_image = NiftiFileLoader(moving_image_dir, load_into_memory)
        loader_fixed_image = NiftiFileLoader(fixed_image_dir, load_into_memory)
        loader_moving_label = NiftiFileLoader(moving_label_dir, load_into_memory)
        loader_fixed_label = NiftiFileLoader(fixed_label_dir, load_into_memory)

        # sanity check
        # filenames should be the same
        assert loader_moving_image.get_sorted_keys() == loader_fixed_image.get_sorted_keys()
        assert loader_moving_image.get_sorted_keys() == loader_moving_label.get_sorted_keys()
        assert loader_moving_image.get_sorted_keys() == loader_fixed_label.get_sorted_keys()

        moving_image_shape = loader_moving_image.get_image_shape()
        fixed_image_shape = loader_fixed_image.get_image_shape()
        moving_label_shape = loader_moving_label.get_image_shape()
        fixed_label_shape = loader_fixed_label.get_image_shape()

        # sanity check
        # image and label have same shape
        assert moving_image_shape == moving_label_shape
        assert fixed_image_shape == fixed_label_shape

        # save variables
        self.file_names = loader_moving_image.get_sorted_keys()

        self.loader_moving_image = loader_moving_image
        self.loader_fixed_image = loader_fixed_image
        self.loader_moving_label = loader_moving_label
        self.loader_fixed_label = loader_fixed_label

        self.moving_image_shape = moving_image_shape  # [dim1, dim2, dim3]
        self.fixed_image_shape = fixed_image_shape  # [dim1, dim2, dim3]
        self.sample_label = sample_label
        self.num_images = len(self.file_names)
        self.num_indices = 2

    def get_generator(self):
        for image_index, image_key in enumerate(self.file_names):
            moving_image = self.loader_moving_image.get_data(image_key) / 255.
            fixed_image = self.loader_fixed_image.get_data(image_key) / 255.
            moving_label = self.loader_moving_label.get_data(image_key)
            fixed_label = self.loader_fixed_label.get_data(image_key)

            if len(moving_label.shape) == 4:  # multiple labels
                label_indices = get_label_indices(moving_label.shape[3], self.sample_label)
                for label_index in label_indices:
                    indices = np.asarray([image_index, label_index], dtype=np.float32)
                    yield (moving_image, fixed_image, moving_label[..., label_index], indices), \
                          fixed_label[..., label_index]
            elif len(moving_label.shape) == 3:  # only one label
                label_index = 0
                indices = np.asarray([image_index, label_index], dtype=np.float32)
                yield (moving_image, fixed_image, moving_label, indices), fixed_label
            else:
                raise ValueError("Unknown moving_label.shape")


def get_fnames_in_dir(dir_name):
    assert dir_name is not None
    file_names = os.listdir(dir_name)
    file_names.sort()
    file_path_names = [os.path.join(dir_name, x) for x in file_names]
    return file_names, file_path_names


def load_from_nifti(x):
    return np.asarray(nib.load(x).dataobj, dtype=np.float32)
