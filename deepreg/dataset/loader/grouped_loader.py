"""
Load grouped data.
Supported formats: h5 and Nifti.
Image data can be labeled or unlabeled.
Read https://deepreg.readthedocs.io/en/latest/api/loader.html#module-deepreg.dataset.loader.grouped_loader for more details.
"""
import random
from copy import deepcopy
from typing import List, Optional, Tuple, Union

from deepreg.dataset.loader.interface import (
    AbstractUnpairedDataLoader,
    GeneratorDataLoader,
)
from deepreg.dataset.util import check_difference_between_two_lists
from deepreg.registry import REGISTRY


@REGISTRY.register_data_loader(name="grouped")
class GroupedDataLoader(AbstractUnpairedDataLoader, GeneratorDataLoader):
    """
    Load grouped data.

    Yield indexes of images to load using
    sample_index_generator from GeneratorDataLoader.
    AbstractUnpairedLoader handles different file formats
    """

    def __init__(
        self,
        file_loader,
        data_dir_paths: List[str],
        labeled: bool,
        sample_label: Optional[str],
        intra_group_prob: float,
        intra_group_option: str,
        sample_image_in_group: bool,
        seed: Optional[int],
        image_shape: Union[Tuple[int, ...], List[int]],
    ):
        """
        :param file_loader: a subclass of FileLoader
        :param data_dir_paths: paths of the directory storing data,
          the data has to be saved under two different sub-directories:

          - images
          - labels

        :param labeled: bool, true if the data is labeled, false if unlabeled
        :param sample_label: "sample" or "all", read `get_label_indices`
            in deepreg/dataset/util.py for more details.
        :param intra_group_prob: float between 0 and 1,

          - 0 means generating only inter-group samples,
          - 1 means generating only intra-group samples

        :param intra_group_option: str, "forward", "backward, or "unconstrained"
        :param sample_image_in_group: bool,

          - if true, only one image pair will be yielded for each group,
            so one epoch has num_groups pairs of data,
          - if false, iterate through this loader will generate all possible pairs

        :param seed: controls the randomness in sampling,
            if seed=None, then the randomness is not fixed
        :param image_shape: list or tuple of length 3,
            corresponding to (dim1, dim2, dim3) of the 3D image
        """
        super().__init__(
            image_shape=image_shape,
            labeled=labeled,
            sample_label=sample_label,
            seed=seed,
        )
        assert isinstance(
            data_dir_paths, list
        ), f"data_dir_paths must be list of strings, got {data_dir_paths}"
        # init
        # the indices for identifying an image pair is (group1, sample1, group2, sample2, label)
        self.num_indices = 5
        self.intra_group_option = intra_group_option
        self.intra_group_prob = intra_group_prob
        self.sample_image_in_group = sample_image_in_group
        # set file loaders
        # grouped data are not paired data, so moving/fixed share the same file loader for images/labels
        loader_image = file_loader(
            dir_paths=data_dir_paths, name="images", grouped=True
        )
        self.loader_moving_image = loader_image
        self.loader_fixed_image = loader_image
        if self.labeled is True:
            loader_label = file_loader(
                dir_paths=data_dir_paths, name="labels", grouped=True
            )
            self.loader_moving_label = loader_label
            self.loader_fixed_label = loader_label
        self.validate_data_files()
        # get group related stats
        self.num_groups = self.loader_moving_image.get_num_groups()
        self.num_images_per_group = self.loader_moving_image.get_num_images_per_group()
        if self.intra_group_prob < 1:
            if self.num_groups < 2:
                raise ValueError(
                    f"There are {self.num_groups} groups, "
                    f"we need at least two groups for inter group sampling"
                )
        # calculate number of samples and save pre-calculated sample indices
        if self.sample_image_in_group is True:
            # one image pair in each group (pair) will be yielded
            self.sample_indices = None
            self._num_samples = self.num_groups
        else:
            # all possible pair in each group (pair) will be yielded
            if intra_group_prob not in [0, 1]:
                raise ValueError(
                    "Mixing intra and inter groups is not supported"
                    " when not sampling pairs."
                )
            if intra_group_prob == 0:  # inter group
                self.sample_indices = self.get_inter_sample_indices()
            else:  # intra group
                self.sample_indices = self.get_intra_sample_indices()

            self._num_samples = len(self.sample_indices)  # type: ignore

    def validate_data_files(self):
        """If the data are labeled, verify image loader and label loader have the same files."""
        if self.labeled is True:
            image_ids = self.loader_moving_image.get_data_ids()
            label_ids = self.loader_moving_label.get_data_ids()
            check_difference_between_two_lists(
                list1=image_ids,
                list2=label_ids,
                name="images and labels in grouped loader",
            )

    def get_intra_sample_indices(self) -> list:
        """
        Calculate the sample indices for intra-group sampling
        The index to identify a sample is (group1, image1, group2, image2), means
        - image1 of group1 is moving image
        - image2 of group2 is fixed image

        Assuming group i has ni images,
        then in total the number of samples are
        - sum( ni * (ni-1) / 2 ) for forward/backward
        - sum( ni * (ni-1) ) for unconstrained

        :return: a list of sample indices
        """
        intra_sample_indices = []
        for group_index in range(self.num_groups):
            num_images_in_group = self.num_images_per_group[group_index]
            if self.intra_group_option == "forward":
                for i in range(num_images_in_group):
                    for j in range(i):
                        # j < i
                        intra_sample_indices.append((group_index, j, group_index, i))
            elif self.intra_group_option == "backward":
                for i in range(num_images_in_group):
                    for j in range(i):
                        # i > j
                        intra_sample_indices.append((group_index, i, group_index, j))
            elif self.intra_group_option == "unconstrained":
                for i in range(num_images_in_group):
                    for j in range(i):
                        # j < i, i > j
                        intra_sample_indices.append((group_index, j, group_index, i))
                        intra_sample_indices.append((group_index, i, group_index, j))
            else:
                raise ValueError(
                    "Unknown intra_group_option, must be forward/backward/unconstrained"
                )
        return intra_sample_indices

    def get_inter_sample_indices(self) -> list:
        """
        Calculate the sample indices for inter-group sampling
        The index to identify a sample is (group1, image1, group2, image2), means

          - image1 of group1 is moving image
          - image2 of group2 is fixed image

        All pairs of images in the dataset are registered.
        Assuming group i has ni images and that N=[n1, n2, ..., nI],
        then in total the number of samples are:
        sum(N) * (sum(N)-1) - sum( N * (N-1) )

        :return: a list of sample indices
        """
        inter_sample_indices = []
        for group_index1 in range(self.num_groups):
            for group_index2 in range(self.num_groups):
                if group_index1 == group_index2:  # do not sample from the same group
                    continue
                num_images_in_group1 = self.num_images_per_group[group_index1]
                num_images_in_group2 = self.num_images_per_group[group_index2]
                for image_index1 in range(num_images_in_group1):
                    for image_index2 in range(num_images_in_group2):
                        inter_sample_indices.append(
                            (group_index1, image_index1, group_index2, image_index2)
                        )
        return inter_sample_indices

    def sample_index_generator(self):
        """
        Yield (moving_index, fixed_index, image_indices) sequentially, where

          - moving_index = (group1, image1)
          - fixed_index = (group2, image2)
          - image_indices = [group1, image1, group2, image2]
        """
        rnd = random.Random(self.seed)  # set random seed
        if self.sample_image_in_group is True:
            # for each group sample one image pair only
            group_indices = [i for i in range(self.num_groups)]
            rnd.shuffle(group_indices)
            for group_index in group_indices:
                if rnd.random() <= self.intra_group_prob:
                    # intra-group sampling
                    # inside the group_index-th group, we sample two images as moving/fixed
                    group_index1 = group_index
                    group_index2 = group_index
                    num_images_in_group = self.num_images_per_group[group_index]
                    if num_images_in_group < 2:
                        # skip groups having <2 images
                        # currently have not encountered
                        continue  # pragma: no cover

                    image_index1, image_index2 = rnd.sample(
                        [i for i in range(num_images_in_group)], 2
                    )  # sample two unique indices
                    if self.intra_group_option == "forward":
                        # image_index1 < image_index2
                        image_index1, image_index2 = (
                            min(image_index1, image_index2),
                            max(image_index1, image_index2),
                        )
                    elif self.intra_group_option == "backward":
                        # image_index1 > image_index2
                        image_index1, image_index2 = (
                            max(image_index1, image_index2),
                            min(image_index1, image_index2),
                        )
                    elif self.intra_group_option == "unconstrained":
                        pass
                    else:
                        raise ValueError(
                            f"Unknown intra_group_option, "
                            f"must be forward/backward/unconstrained, "
                            f"got {self.intra_group_option}"
                        )
                else:
                    # inter-group sampling
                    # we sample another group, then in each group we sample one image
                    group_index1 = group_index
                    group_index2 = rnd.choice(
                        [i for i in range(self.num_groups) if i != group_index]
                    )
                    num_images_in_group1 = self.num_images_per_group[group_index1]
                    num_images_in_group2 = self.num_images_per_group[group_index2]
                    image_index1 = rnd.choice([i for i in range(num_images_in_group1)])
                    image_index2 = rnd.choice([i for i in range(num_images_in_group2)])

                moving_index = (group_index1, image_index1)
                fixed_index = (group_index2, image_index2)
                image_indices = [group_index1, image_index1, group_index2, image_index2]
                yield moving_index, fixed_index, image_indices
        else:
            # sample indices are pre-calculated
            assert self.sample_indices is not None
            sample_indices = deepcopy(self.sample_indices)
            rnd.shuffle(sample_indices)  # shuffle in place
            for sample_index in sample_indices:
                group_index1, image_index1, group_index2, image_index2 = sample_index
                moving_index = (group_index1, image_index1)
                fixed_index = (group_index2, image_index2)
                image_indices = [group_index1, image_index1, group_index2, image_index2]
                yield moving_index, fixed_index, image_indices

    def close(self):
        """Close file loaders"""
        self.loader_moving_image.close()
        if self.labeled is True:
            self.loader_moving_label.close()
