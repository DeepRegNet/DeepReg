'''
Loads paired, unlabeled h5 data
The data must be arranged in directories with names:
images
'''

import os
import random

from deepreg.data.loader import UnpairedDataLoader, GeneratorDataLoader
from deepreg.data.h5.util import H5FileLoader


class H5UnpairedUnlabeledDataLoader(UnpairedDataLoader, GeneratorDataLoader):
    '''
    This class loads unpaired, unlabeled h5 data
    '''
    def __init__(self, data_dir_path: str, sample_label: str, 
                 seed, image_shape: (list, tuple)):
        """
        Load data which are unpaired and unlabeled, so each sample has
            (image)

        :param data_dir_path: path of the directory storing data, 
        the data has to be saved under different
        sub-directories: images
        :param sample_label:
        :param seed:
        :param image_shape: (width, height, depth)
        """
        super(H5UnpairedUnlabeledDataLoader, self).__init__(image_shape=image_shape,
                                                            sample_label=sample_label,
                                                            seed=seed)
        self.loader_image = H5FileLoader(os.path.join(data_dir_path, "images"))

        self.num_images = len(self.loader_image.get_data_names())
        self._num_samples = self.num_images // 2
        self.labeled = False

    def get_generator(self):
        image_indices = [i for i in range(self.num_images)]
        random.Random(self.seed).shuffle(image_indices)
        for sample_index in range(self.num_samples):
            image_index1 = 2 * sample_index
            image_index2 = 2 * sample_index + 1
            moving_image = self.loader_image.get_data(index=image_index1) / 255.
            fixed_image = self.loader_image.get_data(index=image_index2) / 255.
            moving_label = None
            fixed_label = None

            for sample in self.sample_image_label(moving_image=moving_image,
                                                  fixed_image=fixed_image,
                                                  moving_label=moving_label,
                                                  fixed_label=fixed_label,
                                                  image_indices=[image_index1, image_index2]):
                yield sample    




