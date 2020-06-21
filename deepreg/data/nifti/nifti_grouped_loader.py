import os
import random

from deepreg.data.loader import UnpairedDataLoader, GeneratorDataLoader
from deepreg.data.nifti.nifti_loader import NiftiFileLoader
from deepreg.data.util import check_difference_between_two_lists


class NiftiGroupedDataLoader(UnpairedDataLoader, GeneratorDataLoader):
    def __init__(self,
                 data_dir_path: str, labeled: bool, sample_label: str,
                 intra_group_prob: float, intra_group_option: str,
                 seed, image_shape: (list, tuple)):
        """
        Load data which are grouped, labeled or unlabeled

        :param data_dir_path: path of the directory storing data,  the data has to be saved under four different
                              sub-directories: images, labels
        :param sample_label:
        :param seed:
        :param image_shape: (width, height, depth)
        """
        super(NiftiGroupedDataLoader, self).__init__(image_shape=image_shape,
                                                     labeled=labeled,
                                                     sample_label=sample_label,
                                                     seed=seed)
        self.num_indices = 5  # (group1, sample1, group2, sample2, label)
        loader_image = NiftiFileLoader(os.path.join(data_dir_path, "images"), grouped=True)
        self.loader_moving_image = loader_image
        self.loader_fixed_image = loader_image
        if self.labeled:
            loader_label = NiftiFileLoader(os.path.join(data_dir_path, "labels"), grouped=True)
            self.loader_moving_label = loader_label
            self.loader_fixed_label = loader_label
        self.validate_data_files()

        self.num_groups = len(self.loader_moving_image.group_paths)
        self.num_images_per_group = [len(self.loader_moving_image.file_path_dict[g])
                                     for g in self.loader_moving_image.group_paths]
        self.intra_sample_indices, self.inter_sample_indices = self.get_sample_indices(intra_group_option)
        self.intra_group_prob = intra_group_prob
        if intra_group_prob == 0:  # inter only
            self._num_samples = len(self.inter_sample_indices)
        elif intra_group_prob == 1:  # intra only
            self._num_samples = len(self.intra_sample_indices)
        else:  # mixed
            if intra_group_prob < 0 or intra_group_prob > 1:
                raise ValueError("intra_group_prob should be between [0,1]")
            # TODO this value is too large
            self._num_samples = len(self.inter_sample_indices) + len(self.intra_sample_indices)

    def validate_data_files(self):
        """Verify all loader have the same files"""
        if self.labeled:
            filenames_image = self.loader_moving_image.get_relative_file_paths()
            filenames_label = self.loader_moving_label.get_relative_file_paths()
            check_difference_between_two_lists(list1=filenames_image, list2=filenames_label)

    def get_sample_indices(self, intra_group_option):
        """
        Set the sample indices for inter/intra group
        One sample index is ((group1, image1), (group2, image2)), where
        - image1 of group1 is moving image
        - image2 of group2 is fixed image
        :return:
        """

        # intra group
        # assuming group i has ni images
        # then in total there are at most sum(ni ** 2) intra samples
        intra_sample_indices = []
        for group_index in range(self.num_groups):
            num_images_in_group = self.num_images_per_group[group_index]
            if intra_group_option == "forward":
                for i in range(num_images_in_group):
                    for j in range(i):
                        # j < i
                        intra_sample_indices.append(((group_index, j), (group_index, i)))
            elif intra_group_option == "backward":
                for i in range(num_images_in_group):
                    for j in range(i):
                        # i > j
                        intra_sample_indices.append(((group_index, i), (group_index, j)))
            elif intra_group_option == "bidirectional":
                for i in range(num_images_in_group):
                    for j in range(i):
                        # j < i, i > j
                        intra_sample_indices.append(((group_index, j), (group_index, i)))
                        intra_sample_indices.append(((group_index, i), (group_index, j)))
            elif intra_group_option == "all":
                for i in range(num_images_in_group):
                    for j in range(num_images_in_group):
                        intra_sample_indices.append(((group_index, i), (group_index, j)))
            else:
                raise ValueError("Unknown intra_group_option, must be forward/backward/bidirectional/all")

        # inter group
        # assuming group i has ni images
        # then in total there are sum(ni) ** 2 - sum(ni ** 2) inter samples
        inter_sample_indices = []
        for group_index1 in range(self.num_groups):
            for group_index2 in range(self.num_groups):
                num_images_in_group1 = self.num_images_per_group[group_index1]
                num_images_in_group2 = self.num_images_per_group[group_index2]
                for image_index1 in range(num_images_in_group1):
                    for image_index2 in range(num_images_in_group2):
                        inter_sample_indices.append(((group_index1, image_index1), (group_index2, image_index2)))

        return intra_sample_indices, inter_sample_indices

    def sample_index_generator(self):
        rnd = random.Random(self.seed)
        if self.intra_group_prob == 0 or self.intra_group_prob == 1:  # inter or intra only
            sample_indices = self.intra_sample_indices.copy() if self.intra_group_prob == 1 \
                else self.inter_sample_indices.copy()
            rnd.shuffle(sample_indices)
            for sample_index in sample_indices:
                moving_index, fixed_index = sample_index
                yield moving_index, fixed_index, list(moving_index) + list(fixed_index)
        else:  # mixed
            intra_sample_indices = self.intra_sample_indices.copy()
            inter_sample_indices = self.inter_sample_indices.copy()
            rnd.shuffle(intra_sample_indices)
            rnd.shuffle(inter_sample_indices)
            while len(intra_sample_indices) > 0 and len(inter_sample_indices) > 0:
                if rnd.random() <= self.intra_group_prob:  # intra group
                    sample_index = intra_sample_indices.pop(0)
                else:  # inter group
                    sample_index = inter_sample_indices.pop(0)
                moving_index, fixed_index = sample_index
                yield moving_index, fixed_index, list(moving_index) + list(fixed_index)
