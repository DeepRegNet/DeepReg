import os

import nibabel as nib
import numpy as np

from deepreg.data.loader import GeneratorDataLoader
from deepreg.data.nifti.sample import get_label_indices
from deepreg.data.nifti.util import get_sorted_filenames_in_dir, check_difference_between_two_lists


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


class NiftiDataLoader(GeneratorDataLoader):
    def __init__(self,
                 data_dir_path: str, moving_image_shape: (list, tuple), fixed_image_shape: (list, tuple),
                 sample_label):
        """

        :param data_dir_path: path of the directory storing data,  the data has to be saved under four different
                              sub-directories: moving_images, fixed_images, moving_labels, fixed_labels
        :param moving_image_shape: (width, height, depth)
        :param fixed_image_shape: (width, height, depth)
        :param sample_label:
        """
        super(NiftiDataLoader, self).__init__()
        if len(moving_image_shape) != 3 or len(fixed_image_shape) != 3:
            raise ValueError("moving_image_shape and fixed_image_shape have to be length of three,"
                             "corresponding to (width, height, depth)")
        self.loader_moving_image = NiftiFileLoader(os.path.join(data_dir_path, "moving_images"))
        self.loader_fixed_image = NiftiFileLoader(os.path.join(data_dir_path, "fixed_images"))
        self.loader_moving_label = NiftiFileLoader(os.path.join(data_dir_path, "moving_labels"))
        self.loader_fixed_label = NiftiFileLoader(os.path.join(data_dir_path, "fixed_labels"))
        self.validate_data_files()

        self.moving_image_shape = tuple(moving_image_shape)
        self.fixed_image_shape = tuple(fixed_image_shape)

        self.num_images = len(self.loader_moving_image.file_paths)
        self.sample_label = sample_label

    def validate_data_files(self):
        """Verify all loader have the same files"""
        filenames_moving_image = self.loader_moving_image.get_relative_file_paths()
        filenames_fixed_image = self.loader_fixed_image.get_relative_file_paths()
        filenames_moving_label = self.loader_moving_label.get_relative_file_paths()
        filenames_fixed_label = self.loader_fixed_label.get_relative_file_paths()
        check_difference_between_two_lists(list1=filenames_moving_image, list2=filenames_fixed_image)
        check_difference_between_two_lists(list1=filenames_moving_image, list2=filenames_moving_label)
        check_difference_between_two_lists(list1=filenames_moving_image, list2=filenames_fixed_label)

    def validate_images_and_labels(self, index: int, moving_image: np.ndarray, fixed_image: np.ndarray,
                                   moving_label: np.ndarray, fixed_label: np.ndarray):
        for arr, name in zip([moving_image, fixed_image, moving_label, fixed_label],
                             ["moving_image", "fixed_image", "moving_label", "fixed_label"]):
            if np.min(arr) < 0 or np.max(arr) > 1:
                raise ValueError("Sample {}'s {} has value outside of [0,1]."
                                 "Images are assumed to be between [0, 255] "
                                 "and labels are assumed to be between [0, 1]".format(index, name))
        for arr, name in zip([moving_image, moving_label],
                             ["moving_image", "moving_label"]):
            if arr.shape[:3] != self.moving_image_shape:
                raise ValueError("Sample {}'s {} has different shape (width, height, depth) from required."
                                 "Expected {} but got {}.".format(index, name, self.moving_image_shape, arr.shape[:3]))
        for arr, name in zip([fixed_image, fixed_label],
                             ["fixed_image", "fixed_label"]):
            if arr.shape[:3] != self.fixed_image_shape:
                raise ValueError("Sample {}'s {} has different shape (width, height, depth) from required."
                                 "Expected {} but got {}.".format(index, name, self.fixed_image_shape, arr.shape[:3]))

    def get_generator(self):
        for image_index in range(self.num_images):
            moving_image = self.loader_moving_image.get_data(index=image_index) / 255.
            fixed_image = self.loader_fixed_image.get_data(index=image_index) / 255.
            moving_label = self.loader_moving_label.get_data(index=image_index)
            fixed_label = self.loader_fixed_label.get_data(index=image_index)
            self.validate_images_and_labels(image_index, moving_image, fixed_image, moving_label, fixed_label)

            if len(moving_label.shape) == 4:  # multiple labels
                label_indices = get_label_indices(moving_label.shape[3], self.sample_label)
                for label_index in label_indices:
                    indices = np.asarray([image_index, label_index], dtype=np.float32)
                    inputs = (moving_image, fixed_image, moving_label[..., label_index], indices)
                    labels = fixed_label[..., label_index]
                    yield inputs, labels
            elif len(moving_label.shape) == 3:  # only one label
                label_index = 0
                indices = np.asarray([image_index, label_index], dtype=np.float32)
                inputs = (moving_image, fixed_image, moving_label, indices)
                labels = fixed_label
                yield inputs, labels
            else:
                raise ValueError("Unknown moving_label.shape")
