'''
Loads paired, unlabeled h5 data
The data must be arranged in directories with names:
moving_image, fixed_image
'''

import os

from deepreg.data.loader import PairedDataLoader, GeneratorDataLoader
from deepreg.data.h5.util import H5FileLoader
from deepreg.data.util import check_difference_between_two_lists


class H5PairedUnlabeledDataLoader(PairedDataLoader, GeneratorDataLoader):
    '''
    This class loads paired, unlabeled h5 data
    '''
    def __init__(self, data_dir_path: str, sample_label: str,
                 seed, moving_image_shape: (list, tuple), fixed_image_shape):
        """
        Load data which are paired and labeled, so each sample has
            (moving_image, fixed_image)

        :param data_dir_path: path of the directory storing data, 
        the data has to be saved under different
        sub-directories: moving_images, fixed_images
        :param sample_label:
        :param seed:
        :param moving_image_shape: (width, height, depth)
        :param fixed_image_shape: (width, height, depth)
        """
        super(H5PairedUnlabeledDataLoader, self).__init__(moving_image_shape=moving_image_shape,
                                                          fixed_image_shape=fixed_image_shape,
                                                          sample_label=sample_label, seed=seed)
        
        self.loader_moving_image = H5FileLoader(os.path.join(data_dir_path, "moving_images"))
        self.loader_fixed_image = H5FileLoader(os.path.join(data_dir_path, "fixed_images"))

        self.num_images = len(self.loader_moving_image.get_data_names())
        self.labeled = False
        self.validate_data_files()
        
        
    def validate_data_files(self):
        '''
        Check that all data names are the same in all folders
        '''
        data_names_moving_image = self.loader_moving_image.get_data_names()
        data_names_fixed_image = self.loader_fixed_image.get_data_names()
        
        check_difference_between_two_lists(list1=data_names_moving_image, 
                                           list2=data_names_fixed_image)
   
    def get_generator(self):
        for image_index in range(self.num_images):
            moving_image = self.loader_moving_image.get_data(index=image_index) / 255.
            fixed_image = self.loader_fixed_image.get_data(index=image_index) / 255.
            moving_label = None
            fixed_label = None

            for sample in self.sample_image_label(moving_image=moving_image,
                                                  fixed_image=fixed_image,
                                                  moving_label=moving_label,
                                                  fixed_label=fixed_label,
                                                  image_indices=[image_index]):
                yield sample

        


