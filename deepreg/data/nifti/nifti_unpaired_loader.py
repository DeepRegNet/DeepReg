import os
import random

from deepreg.data.loader import UnpairedDataLoader, GeneratorDataLoader
from deepreg.data.nifti.nifti_loader import NiftiFileLoader
from deepreg.data.util import check_difference_between_two_lists


class NiftiUnpairedDataLoader(UnpairedDataLoader, GeneratorDataLoader):
    def __init__(self,
                 data_dir_path: str, labeled: bool, sample_label: str, seed, image_shape: (list, tuple)):
        """
        Load data which are unpaired, labeled or unlabeled

        :param data_dir_path: path of the directory storing data,  the data has to be saved under four different
                              sub-directories: images, labels
        :param sample_label:
        :param seed:
        :param image_shape: (width, height, depth)
        """
        super(NiftiUnpairedDataLoader, self).__init__(image_shape=image_shape,
                                                      labeled=labeled,
                                                      sample_label=sample_label,
                                                      seed=seed)
        loader_image = NiftiFileLoader(os.path.join(data_dir_path, "images"), grouped=False)
        self.loader_moving_image = loader_image
        self.loader_fixed_image = loader_image
        if self.labeled:
            loader_label = NiftiFileLoader(os.path.join(data_dir_path, "labels"), grouped=False)
            self.loader_moving_label = loader_label
            self.loader_fixed_label = loader_label
        self.validate_data_files()

        self.num_images = len(self.loader_moving_image.file_paths)
        self._num_samples = self.num_images // 2

    def validate_data_files(self):
        """Verify all loader have the same files"""
        if self.labeled:
            filenames_image = self.loader_moving_image.get_relative_file_paths()
            filenames_label = self.loader_moving_label.get_relative_file_paths()
            check_difference_between_two_lists(list1=filenames_image, list2=filenames_label)

    def sample_index_generator(self):
        image_indices = [i for i in range(self.num_images)]
        random.Random(self.seed).shuffle(image_indices)
        for sample_index in range(self.num_samples):
            moving_index, fixed_index = 2 * sample_index, 2 * sample_index + 1
            yield moving_index, fixed_index, [moving_index, fixed_index]
