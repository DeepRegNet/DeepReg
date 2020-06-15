import os
import random

from deepreg.data.loader import UnpairedDataLoader, GeneratorDataLoader
from deepreg.data.nifti.util import NiftiFileLoader
from deepreg.data.util import check_difference_between_two_lists


class NiftiUnpairedDataLoader(UnpairedDataLoader, GeneratorDataLoader):
    def __init__(self,
                 data_dir_path: str, sample_label: str, seed, image_shape: (list, tuple)):
        """
        :param data_dir_path: path of the directory storing data,  the data has to be saved under four different
                              sub-directories: images, labels
        :param sample_label:
        :param seed:
        :param image_shape: (width, height, depth)
        """
        super(NiftiUnpairedDataLoader, self).__init__(image_shape=image_shape, sample_label=sample_label, seed=seed)
        self.loader_image = NiftiFileLoader(os.path.join(data_dir_path, "images"))
        self.loader_label = NiftiFileLoader(os.path.join(data_dir_path, "labels"))
        self.validate_data_files()

        self.num_images = len(self.loader_image.file_paths)
        self._num_samples = self.num_images // 2

    def validate_data_files(self):
        """Verify all loader have the same files"""
        filenames_image = self.loader_image.get_relative_file_paths()
        filenames_label = self.loader_label.get_relative_file_paths()
        check_difference_between_two_lists(list1=filenames_image, list2=filenames_label)

    @property
    def num_samples(self) -> int:
        return self._num_samples

    def get_generator(self):
        image_indices = [i for i in range(self.num_images)]
        random.Random(self.seed).shuffle(image_indices)
        for sample_index in range(self.num_samples):
            image_index1 = 2 * sample_index
            image_index2 = 2 * sample_index + 1
            moving_image = self.loader_image.get_data(index=image_index1) / 255.
            fixed_image = self.loader_image.get_data(index=image_index2) / 255.
            moving_label = self.loader_label.get_data(index=image_index1)
            fixed_label = self.loader_label.get_data(index=image_index2)

            for sample in self.sample_image_label(moving_image=moving_image,
                                                  fixed_image=fixed_image,
                                                  moving_label=moving_label,
                                                  fixed_label=fixed_label,
                                                  image_indices=[image_index1, image_index2]):
                yield sample
