"""
Load paired image data.
Supported formats: h5 and Nifti.
Image data can be labeled or unlabeled.
"""
import random
from typing import List, Tuple, Union

from deepreg.dataset.loader.interface import (
    AbstractPairedDataLoader,
    GeneratorDataLoader,
)
from deepreg.dataset.util import check_difference_between_two_lists
from deepreg.registry import REGISTRY


@REGISTRY.register_data_loader(name="paired")
class PairedDataLoader(AbstractPairedDataLoader, GeneratorDataLoader):
    """
    Load paired data using given file loader.
    The function sample_index_generator needs to be defined for the
    GeneratorDataLoader class.
    """

    def __init__(
        self,
        file_loader,
        data_dir_paths: List[str],
        labeled: bool,
        sample_label: str,
        seed,
        moving_image_shape: Union[Tuple[int, ...], List[int]],
        fixed_image_shape: Union[Tuple[int, ...], List[int]],
    ):
        """
        :param file_loader:
        :param data_dir_paths: path of the directories storing data,
          the data has to be saved under four different
          sub-directories: moving_images, fixed_images, moving_labels,
          fixed_labels
        :param labeled: true if the data are labeled
        :param sample_label:
        :param seed:
        :param moving_image_shape: (width, height, depth)
        :param fixed_image_shape: (width, height, depth)
        """
        super().__init__(
            moving_image_shape=moving_image_shape,
            fixed_image_shape=fixed_image_shape,
            labeled=labeled,
            sample_label=sample_label,
            seed=seed,
        )
        assert isinstance(
            data_dir_paths, list
        ), f"data_dir_paths must be list of strings, got {data_dir_paths}"

        for ddp in data_dir_paths:
            assert isinstance(
                ddp, str
            ), f"data_dir_paths must be list of strings, got {data_dir_paths}"

        self.loader_moving_image = file_loader(
            dir_paths=data_dir_paths, name="moving_images", grouped=False
        )
        self.loader_fixed_image = file_loader(
            dir_paths=data_dir_paths, name="fixed_images", grouped=False
        )
        if self.labeled:
            self.loader_moving_label = file_loader(
                dir_paths=data_dir_paths, name="moving_labels", grouped=False
            )
            self.loader_fixed_label = file_loader(
                dir_paths=data_dir_paths, name="fixed_labels", grouped=False
            )
        self.validate_data_files()
        self.num_images = self.loader_moving_image.get_num_images()

    def validate_data_files(self):
        """Verify all loaders have the same files."""
        moving_image_ids = self.loader_moving_image.get_data_ids()
        fixed_image_ids = self.loader_fixed_image.get_data_ids()
        check_difference_between_two_lists(
            list1=moving_image_ids,
            list2=fixed_image_ids,
            name="moving and fixed images in paired loader",
        )
        if self.labeled:
            moving_label_ids = self.loader_moving_label.get_data_ids()
            fixed_label_ids = self.loader_fixed_label.get_data_ids()
            check_difference_between_two_lists(
                list1=moving_image_ids,
                list2=moving_label_ids,
                name="moving images and labels in paired loader",
            )
            check_difference_between_two_lists(
                list1=moving_image_ids,
                list2=fixed_label_ids,
                name="fixed images and labels in paired loader",
            )

    def sample_index_generator(self):
        """
        Generate indexes in order to load data using the
        GeneratorDataLoader class.
        """
        image_indices = [i for i in range(self.num_images)]
        random.Random(self.seed).shuffle(image_indices)
        for image_index in image_indices:
            yield image_index, image_index, [image_index]

    def close(self):
        self.loader_moving_image.close()
        self.loader_fixed_image.close()
        if self.labeled:
            self.loader_moving_label.close()
            self.loader_fixed_label.close()
