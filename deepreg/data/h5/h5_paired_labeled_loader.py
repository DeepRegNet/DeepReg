import os

from deepreg.data.loader import PairedDataLoader, GeneratorDataLoader
from deepreg.data.h5.util import H5FileLoader
from deepreg.data.util import check_difference_between_two_lists


class H5PairedLabeledDataLoader(PairedDataLoader, GeneratorDataLoader):
    
    def __init__(self, data_dir_path: str, sample_label: str, seed, moving_image_shape: (list, tuple), fixed_image_shape):
        """
        Load data which are paired and labeled, so each sample has
            (moving_image, fixed_image, moving_label, fixed_label)

        :param data_dir_path: path of the directory storing data,  the data has to be saved under four different
                              sub-directories: moving_images, fixed_images, moving_labels, fixed_labels
        :param sample_label:
        :param seed:
        :param moving_image_shape: (width, height, depth)
        :param fixed_image_shape: (width, height, depth)
        """
        super(H5PairedLabeledDataLoader, self).__init__(moving_image_shape=moving_image_shape,
                                                           fixed_image_shape=fixed_image_shape,
                                                           sample_label=sample_label, seed=seed)
        
        self.loader_moving_image = H5FileLoader(os.path.join(data_dir_path, "moving_images"))
        self.loader_fixed_image = H5FileLoader(os.path.join(data_dir_path, "fixed_images"))
        self.loader_moving_label = H5FileLoader(os.path.join(data_dir_path, "moving_labels"))
        self.loader_fixed_label = H5FileLoader(os.path.join(data_dir_path, "fixed_labels"))
        
        self.num_images = len(self.loader_moving_image.get_data_names())
        self.validate_data_files()
        
        
    def validate_data_files(self):
        '''
        Check that all data names are the same in all folders
        '''
        data_names_moving_image = self.loader_moving_image.get_data_names()
        data_names_fixed_image = self.loader_fixed_image.get_data_names()
        data_names_moving_label = self.loader_moving_label.get_data_names()
        data_names_fixed_label = self.loader_fixed_label.get_data_names()
        
        check_difference_between_two_lists(list1=data_names_moving_image, list2=data_names_fixed_image)
        check_difference_between_two_lists(list1=data_names_moving_image, list2=data_names_moving_label)
        check_difference_between_two_lists(list1=data_names_moving_image, list2=data_names_fixed_label)
        
    
    def get_generator(self):
        for image_index in range(self.num_images):
            moving_image = self.loader_moving_image.get_data(index=image_index) / 255.
            fixed_image = self.loader_fixed_image.get_data(index=image_index) / 255.
            moving_label = self.loader_moving_label.get_data(index=image_index)
            fixed_label = self.loader_fixed_label.get_data(index=image_index) 

            for sample in self.sample_image_label(moving_image=moving_image,
                                                  fixed_image=fixed_image,
                                                  moving_label=moving_label,
                                                  fixed_label=fixed_label,
                                                  image_indices=[image_index]):
                yield sample

        





# data_dir_path = '/home/ssd/Desktop/DeepReg_Project/DeepReg_Scripts/mr_us/paired/train'


# data_names_moving_image = loader_moving_image.get_data_names()
# data_names_fixed_image = loader_fixed_image.get_data_names()
# data_names_moving_label = loader_moving_label.get_data_names()
# data_names_fixed_label = loader_fixed_label.get_data_names()

# check_difference_between_two_lists(list1=data_names_moving_image, list2=data_names_fixed_image)
# check_difference_between_two_lists(list1=data_names_moving_image, list2=data_names_moving_label)
# check_difference_between_two_lists(list1=data_names_moving_image, list2=data_names_fixed_label)


