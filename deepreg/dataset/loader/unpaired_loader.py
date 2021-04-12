"""
Load unpaired data.
Supported formats: h5 and Nifti.
Image data can be labeled or unlabeled.
"""
import random
from typing import List, Tuple, Union

from deepreg.dataset.loader.interface import (
    AbstractUnpairedDataLoader,
    GeneratorDataLoader,
)
from deepreg.dataset.util import check_difference_between_two_lists
from deepreg.registry import REGISTRY


@REGISTRY.register_data_loader(name="unpaired")
class UnpairedDataLoader(AbstractUnpairedDataLoader, GeneratorDataLoader):
    """
    Load unpaired data using given file loader. Handles both labeled
    and unlabeled cases.
    The function sample_index_generator needs to be defined for the
    GeneratorDataLoader class.
    """

    def __init__(
        self,
        file_loader,
        data_dir_paths: List[str],
        labeled: bool,
        sample_label: str,
        seed: int,
        image_shape: Union[Tuple[int, ...], List[int]],
    ):
        """
        Load data which are unpaired, labeled or unlabeled.

        :param file_loader:
        :param data_dir_paths: paths of the directories storing data,
            the data are saved under four different sub-directories: images, labels
        :param labeled: whether the data is labeled.
        :param sample_label:
        :param seed:
        :param image_shape: (width, height, depth)
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
        loader_image = file_loader(
            dir_paths=data_dir_paths, name="images", grouped=False
        )
        self.loader_moving_image = loader_image
        self.loader_fixed_image = loader_image
        if self.labeled:
            loader_label = file_loader(
                dir_paths=data_dir_paths, name="labels", grouped=False
            )
            self.loader_moving_label = loader_label
            self.loader_fixed_label = loader_label
        self.validate_data_files()

        self.num_images = self.loader_moving_image.get_num_images()
        self._num_samples = self.num_images // 2

    def validate_data_files(self):
        """
        Verify all loader have the same files.
        Since fixed and moving loaders come from the same file_loader,
        there is no need to check both (avoid duplicate).
        """
        if self.labeled:
            image_ids = self.loader_moving_image.get_data_ids()
            label_ids = self.loader_moving_label.get_data_ids()
            check_difference_between_two_lists(
                list1=image_ids,
                list2=label_ids,
                name="images and labels in unpaired loader",
            )

    def sample_index_generator(self):
        """
        Generates sample indexes to load data using the
        GeneratorDataLoader class.
        """
        image_indices = [i for i in range(self.num_images)]
        random.Random(self.seed).shuffle(image_indices)
        for sample_index in range(self.num_samples):
            moving_index, fixed_index = (
                image_indices[2 * sample_index],
                image_indices[2 * sample_index + 1],
            )
            yield moving_index, fixed_index, [moving_index, fixed_index]

    def close(self):
        """
        Close the moving files opened by the file_loaders.
        """
        self.loader_moving_image.close()
        if self.labeled:
            self.loader_moving_label.close()
