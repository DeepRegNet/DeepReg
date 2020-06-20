import os

from deepreg.data.loader import PairedDataLoader, GeneratorDataLoader
from deepreg.data.nifti.util import NiftiFileLoader
from deepreg.data.util import check_difference_between_two_lists


class NiftiPairedUnlabeledDataLoader(PairedDataLoader, GeneratorDataLoader):
    def __init__(self,
                 data_dir_path: str, sample_label: (str, None), seed,
                 moving_image_shape: (list, tuple), fixed_image_shape: (list, tuple)):
        """
        Load data which are paired and labeled, so each sample has
            (moving_image, fixed_image)

        :param data_dir_path: path of the directory storing data,  the data has to be saved under four different
                              sub-directories: moving_images, fixed_images
        :param sample_label: not used
        :param seed:
        :param moving_image_shape: (width, height, depth)
        :param fixed_image_shape: (width, height, depth)
        """
        super(NiftiPairedUnlabeledDataLoader, self).__init__(moving_image_shape=moving_image_shape,
                                                             fixed_image_shape=fixed_image_shape,
                                                             sample_label=None, seed=seed)
        self.loader_moving_image = NiftiFileLoader(os.path.join(data_dir_path, "moving_images"))
        self.loader_fixed_image = NiftiFileLoader(os.path.join(data_dir_path, "fixed_images"))
        self.validate_data_files()

        self.num_images = len(self.loader_moving_image.file_paths)
        self.labeled = False

    def validate_data_files(self):
        """Verify all loader have the same files"""
        filenames_moving_image = self.loader_moving_image.get_relative_file_paths()
        filenames_fixed_image = self.loader_fixed_image.get_relative_file_paths()
        check_difference_between_two_lists(list1=filenames_moving_image, list2=filenames_fixed_image)

    def get_generator(self):
        for image_index in range(self.num_images):
            moving_image = self.loader_moving_image.get_data(index=image_index) / 255.
            fixed_image = self.loader_fixed_image.get_data(index=image_index) / 255.

            for sample in self.sample_image_label(moving_image=moving_image,
                                                  fixed_image=fixed_image,
                                                  moving_label=None,
                                                  fixed_label=None,
                                                  image_indices=[image_index]):
                yield sample
