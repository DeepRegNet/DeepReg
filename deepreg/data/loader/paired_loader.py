import os

from deepreg.data.loader.interface import PairedDataLoader, GeneratorDataLoader
from deepreg.data.util import check_difference_between_two_lists


class NiftiPairedDataLoader(PairedDataLoader, GeneratorDataLoader):
    """
    Loads paired data using given file loader, handles both labeled and unlabeled cases
    The function sample_index_generator needs to be defined for the GeneratorDataLoader class
    """

    def __init__(self,
                 file_loader,
                 data_dir_path: str, labeled: bool, sample_label: str, seed,
                 moving_image_shape: (list, tuple), fixed_image_shape: (list, tuple)):
        """
        Load data which are paired, labeled or unlabeled.
        :param file_loader:
        :param data_dir_path: path of the directory storing data,  the data has to be saved under four different
                              sub-directories: moving_images, fixed_images, moving_labels, fixed_labels
        :param labeled: true if the data are labeled
        :param sample_label:
        :param seed:
        :param moving_image_shape: (width, height, depth)
        :param fixed_image_shape: (width, height, depth)
        """
        super(NiftiPairedDataLoader, self).__init__(moving_image_shape=moving_image_shape,
                                                    fixed_image_shape=fixed_image_shape,
                                                    labeled=labeled,
                                                    sample_label=sample_label,
                                                    seed=seed)

        self.loader_moving_image = file_loader(os.path.join(data_dir_path, "moving_images"), grouped=False)
        self.loader_fixed_image = file_loader(os.path.join(data_dir_path, "fixed_images"), grouped=False)
        if self.labeled:
            self.loader_moving_label = file_loader(os.path.join(data_dir_path, "moving_labels"), grouped=False)
            self.loader_fixed_label = file_loader(os.path.join(data_dir_path, "fixed_labels"), grouped=False)
        self.validate_data_files()
        self.num_images = self.loader_moving_image.get_num_images()

    def validate_data_files(self):
        """Verify all loader have the same files"""
        moving_image_ids = self.loader_moving_image.get_data_ids()
        fixed_image_ids = self.loader_fixed_image.get_data_ids()
        check_difference_between_two_lists(list1=moving_image_ids,
                                           list2=fixed_image_ids)

        if self.labeled:
            moving_label_ids = self.loader_moving_label.get_data_ids()
            fixed_label_ids = self.loader_fixed_label.get_data_ids()
            check_difference_between_two_lists(list1=moving_image_ids,
                                               list2=moving_label_ids)
            check_difference_between_two_lists(list1=moving_image_ids,
                                               list2=fixed_label_ids)

    def sample_index_generator(self):
        """generates indexes in order to load data using the GeneratorDataLoader class"""
        for image_index in range(self.num_images):
            yield image_index, image_index, [image_index]
