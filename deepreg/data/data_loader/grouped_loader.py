import os
import random

from deepreg.data.loader import UnpairedDataLoader, GeneratorDataLoader
from deepreg.data.util import check_difference_between_two_lists


class NiftiGroupedDataLoader(UnpairedDataLoader, GeneratorDataLoader):
    def __init__(self,
                 file_loader,
                 data_dir_path: str, labeled: bool, sample_label: (str, None),
                 intra_group_prob: float, intra_group_option: str, sample_image_in_group: bool,
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
        loader_image = file_loader(os.path.join(data_dir_path, "images"), grouped=True)
        self.loader_moving_image = loader_image
        self.loader_fixed_image = loader_image
        if self.labeled:
            loader_label = file_loader(os.path.join(data_dir_path, "labels"), grouped=True)
            self.loader_moving_label = loader_label
            self.loader_fixed_label = loader_label
        self.validate_data_files()

        self.num_groups = self.loader_moving_image.get_num_groups()
        self.num_images_per_group = self.loader_moving_image.get_num_images_per_group()
        self.intra_group_option = intra_group_option
        self.intra_group_prob = intra_group_prob
        self.sample_image_in_group = sample_image_in_group
        if self.intra_group_prob < 1:
            if self.num_groups < 2:
                raise ValueError("There are <2 groups, can't do inter group sampling")
        if self.sample_image_in_group:
            # one image pair in each group (pair) will be yielded
            self.sample_indices = None
            self._num_samples = self.num_groups
        else:
            # all possible pair in each group (pair) will be yielded
            if intra_group_prob not in [0, 1]:
                raise ValueError("Mixing intra and inter groups is not supported when not sampling pairs.")
            if intra_group_prob == 0:  # inter group
                sample_indices = self.get_inter_sample_indices()
            else:  # intra group
                sample_indices = self.get_intra_sample_indices()
            self._num_samples = len(sample_indices)

    def validate_data_files(self):
        """Verify all loader have the same files"""
        if self.labeled:
            image_ids = self.loader_moving_image.get_data_ids()
            label_ids = self.loader_moving_label.get_data_ids()
            check_difference_between_two_lists(list1=image_ids, list2=label_ids)

    def get_intra_sample_indices(self):
        """
        Set the sample indices for intra group
        One sample index is ((group1, image1), (group2, image2)), where
        - image1 of group1 is moving image
        - image2 of group2 is fixed image
        assuming group i has ni images
        then in total there are at most sum(ni ** 2) intra samples
        :return:
        """
        intra_sample_indices = []
        for group_index in range(self.num_groups):
            num_images_in_group = self.num_images_per_group[group_index]
            if self.intra_group_option == "forward":
                for i in range(num_images_in_group):
                    for j in range(i):
                        # j < i
                        intra_sample_indices.append(((group_index, j), (group_index, i)))
            elif self.intra_group_option == "backward":
                for i in range(num_images_in_group):
                    for j in range(i):
                        # i > j
                        intra_sample_indices.append(((group_index, i), (group_index, j)))
            elif self.intra_group_option == "bidirectional":
                for i in range(num_images_in_group):
                    for j in range(i):
                        # j < i, i > j
                        intra_sample_indices.append(((group_index, j), (group_index, i)))
                        intra_sample_indices.append(((group_index, i), (group_index, j)))
            else:
                raise ValueError("Unknown intra_group_option, must be forward/backward/bidirectional")
        return intra_sample_indices

    def get_inter_sample_indices(self):
        """
        Set the sample indices for inter group
        One sample index is ((group1, image1), (group2, image2)), where
        - image1 of group1 is moving image
        - image2 of group2 is fixed image
        assuming group i has ni images
        then in total there are sum(ni) ** 2 - sum(ni ** 2) inter samples
        :return:
        """
        inter_sample_indices = []
        for group_index1 in range(self.num_groups):
            for group_index2 in range(self.num_groups):
                num_images_in_group1 = self.num_images_per_group[group_index1]
                num_images_in_group2 = self.num_images_per_group[group_index2]
                for image_index1 in range(num_images_in_group1):
                    for image_index2 in range(num_images_in_group2):
                        inter_sample_indices.append(((group_index1, image_index1), (group_index2, image_index2)))
        return inter_sample_indices

    def sample_index_generator(self):
        rnd = random.Random(self.seed)
        if self.sample_image_in_group:
            group_indices = [i for i in range(self.num_groups)]
            random.Random(self.seed).shuffle(group_indices)
            for group_index in group_indices:
                if rnd.random() <= self.intra_group_prob:  # intra group, inside one group
                    num_images_in_group = self.num_images_per_group[group_index]
                    if self.intra_group_option in ["forward", "backward"]:
                        # image_index1 < image_index2
                        # image_index1 must be <= num_images_in_group-2
                        image_index1 = rnd.choice([i for i in range(num_images_in_group - 1)])
                        image_index2 = rnd.choice([i for i in range(image_index1 + 1, num_images_in_group)])
                        if self.intra_group_option == "forward":
                            yield (group_index, image_index1), (group_index, image_index2), [group_index, image_index1,
                                                                                             group_index, image_index2]
                        else:
                            yield (group_index, image_index2), (group_index, image_index1), [group_index, image_index2,
                                                                                             group_index, image_index1]
                    elif self.intra_group_option == "unconstrained":
                        image_index1, image_index2 = rnd.sample([i for i in range(num_images_in_group)], 2)
                        image_index2 = rnd.choice([i for i in range(num_images_in_group) if i != image_index1])
                        yield (group_index, image_index1), (group_index, image_index2), [group_index, image_index1,
                                                                                         group_index, image_index2]
                    else:
                        raise ValueError("Unknown intra_group_option, must be forward/backward/unconstrained")
                else:  # inter group, between different groups
                    group_index1 = group_index
                    group_index2 = rnd.choice([i for i in range(self.num_groups) if i != group_index1])
                    num_images_in_group1 = self.num_images_per_group[group_index1]
                    num_images_in_group2 = self.num_images_per_group[group_index2]
                    image_index1 = rnd.choice([i for i in range(num_images_in_group1)])
                    image_index2 = rnd.choice([i for i in range(num_images_in_group2)])
                    yield (group_index1, image_index1), (group_index2, image_index2), [group_index1, image_index1,
                                                                                       group_index2, image_index2]
        else:
            assert self.sample_indices is not None
            sample_indices = self.sample_indices.copy()
            rnd.shuffle(sample_indices)
            for sample_index in sample_indices:
                yield sample_index
