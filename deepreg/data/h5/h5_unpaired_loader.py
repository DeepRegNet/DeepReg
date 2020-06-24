"""
Loader for unpaired h5 data
Handles both labeled and unlabeled cases
The h5 files must be in folders:
images, labels
"""

import os
import random

from deepreg.data.h5.h5_loader import H5FileLoader
from deepreg.data.loader import UnpairedDataLoader, GeneratorDataLoader
from deepreg.data.util import check_difference_between_two_lists


class H5UnpairedDataLoader(UnpairedDataLoader, GeneratorDataLoader):
    """
    Loads unpaired h5 data, handles both labeled and unlabeled cases
    The function sample_index_generator needs to be defined for the 
    GeneratorDataLoader class
    Attributes and functions from the H5FileLoader are also used in this class
    """

    def __init__(self,
                 data_dir_path: str, labeled: bool, sample_label: str,
                 seed, image_shape: (list, tuple)):
        """
        Load data which are unpaired, labeled or unlabeled

        :param data_dir_path: path of the directory storing data,  
        the data has to be saved under four different
                              sub-directories: images, labels
        :param sample_label:
        :param seed:
        :param image_shape: (width, height, depth)
        """
        super(H5UnpairedDataLoader, self).__init__(image_shape=image_shape,
                                                   labeled=labeled,
                                                   sample_label=sample_label,
                                                   seed=seed)
        loader_image = H5FileLoader(os.path.join(data_dir_path, "images"),
                                    grouped=False)
        self.loader_moving_image = loader_image
        self.loader_fixed_image = loader_image
        if self.labeled:
            loader_label = H5FileLoader(os.path.join(data_dir_path, "labels"),
                                        grouped=False)
            self.loader_moving_label = loader_label
            self.loader_fixed_label = loader_label
        self.validate_data_files()

        self.num_images = len(self.loader_moving_image.get_data_names())
        self._num_samples = self.num_images // 2

    def validate_data_files(self):
        """Verify all loader have the same files"""
        if self.labeled:
            filenames_image = self.loader_moving_image.get_data_names()
            filenames_label = self.loader_moving_label.get_data_names()
            check_difference_between_two_lists(list1=filenames_image,
                                               list2=filenames_label)

    def sample_index_generator(self):
        """
        generates indexes in order to load data using the GeneratorDataLoader 
        class
        """
        image_indices = [i for i in range(self.num_images)]
        random.Random(self.seed).shuffle(image_indices)
        for sample_index in range(self.num_samples):
            moving_index, fixed_index = 2 * sample_index, 2 * sample_index + 1
            yield moving_index, fixed_index, [moving_index, fixed_index]
