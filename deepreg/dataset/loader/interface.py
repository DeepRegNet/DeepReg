"""
Interface between the data loaders and file loaders.
"""

from abc import ABC
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from deepreg import log
from deepreg.dataset.loader.util import normalize_array
from deepreg.dataset.preprocess import resize_inputs
from deepreg.dataset.util import get_label_indices
from deepreg.registry import REGISTRY

logger = log.get(__name__)


class DataLoader:
    """
    loads data to feed to model.
    """

    def __init__(
        self,
        labeled: Optional[bool],
        num_indices: Optional[int],
        sample_label: Optional[str],
        seed: Optional[int] = None,
    ):
        """
        :param labeled: bool corresponding to labels provided or omitted
        :param num_indices:
        :param sample_label:
        :param seed:
        """
        assert labeled in [
            True,
            False,
            None,
        ], f"labeled must be boolean, True or False or None, got {labeled}"
        assert sample_label in [
            "sample",
            "all",
            None,
        ], f"sample_label must be sample, all or None, got {sample_label}"
        assert (
            num_indices is None or num_indices >= 1
        ), f"num_indices must be int >=1 or None, got {num_indices}"
        assert seed is None or isinstance(
            seed, int
        ), f"seed must be None or int, got {seed}"

        self.labeled = labeled
        self.num_indices = num_indices  # number of indices to identify a sample
        self.sample_label = sample_label
        self.seed = seed  # used for sampling

    @property
    def moving_image_shape(self) -> tuple:
        """
        needs to be defined by user.
        """
        raise NotImplementedError

    @property
    def fixed_image_shape(self) -> tuple:
        """
        needs to be defined by user.
        """
        raise NotImplementedError

    @property
    def num_samples(self) -> int:
        """
        Return the number of samples in the dataset for one epoch
        :return:
        """
        raise NotImplementedError

    def get_dataset(self) -> tf.data.Dataset:
        """
        defined in GeneratorDataLoader.
        """
        raise NotImplementedError

    def get_dataset_and_preprocess(
        self,
        training: bool,
        batch_size: int,
        repeat: bool,
        shuffle_buffer_num_batch: int,
        data_augmentation: Optional[Union[List, Dict]] = None,
        num_parallel_calls: int = tf.data.experimental.AUTOTUNE,
    ) -> tf.data.Dataset:
        """
        Generate tf.data.dataset.

        Reference:

            - https://www.tensorflow.org/guide/data_performance#parallelizing_data_transformation
            - https://www.tensorflow.org/api_docs/python/tf/data/Dataset

        :param training: indicating if it's training or not
        :param batch_size: size of mini batch
        :param repeat: indicating if we need to repeat the dataset
        :param shuffle_buffer_num_batch: when shuffling,
            the shuffle_buffer_size = batch_size * shuffle_buffer_num_batch
        :param repeat: indicating if we need to repeat the dataset
        :param data_augmentation: augmentation config, can be a list of dict or dict.
        :param num_parallel_calls: number elements to process asynchronously in parallel
            during preprocessing, -1 means unlimited, heuristically it should be set to
            the number of CPU cores available. AUTOTUNE=-1 means not limited.
        :returns dataset:
        """

        dataset = self.get_dataset()

        # resize
        dataset = dataset.map(
            lambda x: resize_inputs(
                inputs=x,
                moving_image_size=self.moving_image_shape,
                fixed_image_size=self.fixed_image_shape,
            ),
            num_parallel_calls=num_parallel_calls,
        )

        # shuffle / repeat / batch / preprocess
        if training and shuffle_buffer_num_batch > 0:
            dataset = dataset.shuffle(
                buffer_size=batch_size * shuffle_buffer_num_batch,
                reshuffle_each_iteration=True,
            )
        if repeat:
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size=batch_size, drop_remainder=training)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if training and data_augmentation is not None:
            if isinstance(data_augmentation, dict):
                data_augmentation = [data_augmentation]
            for config in data_augmentation:
                da_fn = REGISTRY.build_data_augmentation(
                    config=config,
                    default_args={
                        "moving_image_size": self.moving_image_shape,
                        "fixed_image_size": self.fixed_image_shape,
                        "batch_size": batch_size,
                    },
                )
                dataset = dataset.map(da_fn, num_parallel_calls=num_parallel_calls)

        return dataset

    def close(self):
        pass


class AbstractPairedDataLoader(DataLoader, ABC):
    """
    Abstract loader for paired data independent of file format.
    """

    def __init__(
        self,
        moving_image_shape: Union[Tuple[int, ...], List[int]],
        fixed_image_shape: Union[Tuple[int, ...], List[int]],
        **kwargs,
    ):
        """
        num_indices = 2 corresponding to (image_index, label_index)
        :param moving_image_shape: (width, height, depth)
        :param fixed_image_shape:  (width, height, depth)
        :param kwargs: additional arguments.
        """
        super().__init__(num_indices=2, **kwargs)
        if len(moving_image_shape) != 3 or len(fixed_image_shape) != 3:
            raise ValueError(
                f"moving_image_shape and fixed_image_shape have length of three, "
                f"corresponding to (width, height, depth), "
                f"got moving_image_shape = {moving_image_shape} "
                f"and fixed_image_shape = {fixed_image_shape}"
            )
        self._moving_image_shape = tuple(moving_image_shape)
        self._fixed_image_shape = tuple(fixed_image_shape)
        self.num_images = None

    @property
    def moving_image_shape(self) -> tuple:
        """
        Return the moving image shape.
        :return: shape of moving image
        """
        return self._moving_image_shape

    @property
    def fixed_image_shape(self) -> tuple:
        """
        Return the fixed image shape.
        :return: shape of fixed image
        """
        return self._fixed_image_shape

    @property
    def num_samples(self) -> int:
        """
        Return the number of samples in the dataset for one epoch.
        :return: number of images
        """
        return self.num_images  # type:ignore


class AbstractUnpairedDataLoader(DataLoader, ABC):
    """
    Abstract loader for unparied data independent of file format.
    """

    def __init__(self, image_shape: Union[Tuple[int, ...], List[int]], **kwargs):
        """
        Init.

        :param image_shape: (dim1, dim2, dim3), for unpaired data,
            moving_image_shape = fixed_image_shape = image_shape
        :param kwargs: additional arguments.
        """
        super().__init__(num_indices=3, **kwargs)
        if len(image_shape) != 3:
            raise ValueError(
                f"image_shape has to be length of three, "
                f"corresponding to (width, height, depth), "
                f"got {image_shape}"
            )
        self.image_shape = tuple(image_shape)
        self._num_samples = None

    @property
    def moving_image_shape(self) -> tuple:
        return self.image_shape

    @property
    def fixed_image_shape(self) -> tuple:
        return self.image_shape

    @property
    def num_samples(self) -> int:
        return self._num_samples  # type:ignore


class GeneratorDataLoader(DataLoader, ABC):
    """
    Load samples by implementing get_dataset from DataLoader.
    """

    def __init__(self, **kwargs):
        """
        Init.

        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)
        self.loader_moving_image = None
        self.loader_fixed_image = None
        self.loader_moving_label = None
        self.loader_fixed_label = None

    def get_dataset(self):
        """
        Return a dataset from the generator.
        """
        if self.labeled:
            return tf.data.Dataset.from_generator(
                generator=self.data_generator,
                output_types=dict(
                    moving_image=tf.float32,
                    fixed_image=tf.float32,
                    moving_label=tf.float32,
                    fixed_label=tf.float32,
                    indices=tf.float32,
                ),
                output_shapes=dict(
                    moving_image=tf.TensorShape([None, None, None]),
                    fixed_image=tf.TensorShape([None, None, None]),
                    moving_label=tf.TensorShape([None, None, None]),
                    fixed_label=tf.TensorShape([None, None, None]),
                    indices=self.num_indices,
                ),
            )
        return tf.data.Dataset.from_generator(
            generator=self.data_generator,
            output_types=dict(
                moving_image=tf.float32, fixed_image=tf.float32, indices=tf.float32
            ),
            output_shapes=dict(
                moving_image=tf.TensorShape([None, None, None]),
                fixed_image=tf.TensorShape([None, None, None]),
                indices=self.num_indices,
            ),
        )

    def data_generator(self):
        """
        Yield samples of data to feed model.
        """
        for (moving_index, fixed_index, image_indices) in self.sample_index_generator():
            moving_image = self.loader_moving_image.get_data(index=moving_index)
            moving_image = normalize_array(moving_image)
            fixed_image = self.loader_fixed_image.get_data(index=fixed_index)
            fixed_image = normalize_array(fixed_image)
            moving_label = (
                self.loader_moving_label.get_data(index=moving_index)
                if self.labeled
                else None
            )
            fixed_label = (
                self.loader_fixed_label.get_data(index=fixed_index)
                if self.labeled
                else None
            )

            for sample in self.sample_image_label(
                moving_image=moving_image,
                fixed_image=fixed_image,
                moving_label=moving_label,
                fixed_label=fixed_label,
                image_indices=image_indices,
            ):
                yield sample

    def sample_index_generator(self):
        """
        Method is defined by the implemented data loaders to yield the sample indexes.
        Only used in data_generator.
        """
        raise NotImplementedError

    @staticmethod
    def validate_images_and_labels(
        moving_image: np.ndarray,
        fixed_image: np.ndarray,
        moving_label: Optional[np.ndarray],
        fixed_label: Optional[np.ndarray],
        image_indices: list,
    ):
        """
        Check file names match according to naming convention.
        Only used in sample_image_label.
        :param moving_image: np.ndarray of shape (m_dim1, m_dim2, m_dim3)
        :param fixed_image: np.ndarray of shape (f_dim1, f_dim2, f_dim3)
        :param moving_label: np.ndarray of shape (m_dim1, m_dim2, m_dim3)
            or (m_dim1, m_dim2, m_dim3, num_labels)
        :param fixed_label: np.ndarray of shape (f_dim1, f_dim2, f_dim3)
            or (f_dim1, f_dim2, f_dim3, num_labels)
        :param image_indices: list
        """
        # images should never be None, and labels should all be non-None or None
        if moving_image is None or fixed_image is None:
            raise ValueError("moving image and fixed image must not be None")
        if (moving_label is None) != (fixed_label is None):
            raise ValueError(
                "moving label and fixed label must be both None or non-None"
            )
        # image and label's values should be between [0, 1]
        for arr, name in zip(
            [moving_image, fixed_image, moving_label, fixed_label],
            ["moving_image", "fixed_image", "moving_label", "fixed_label"],
        ):
            if arr is None:
                continue
            if np.min(arr) < 0 or np.max(arr) > 1:
                raise ValueError(
                    f"Sample {image_indices}'s {name}'s values are not between [0, 1]. "
                    f"Its minimum value is {np.min(arr)} "
                    f"and its maximum value is {np.max(arr)}.\n"
                    f"The images are automatically normalized on image level: "
                    f"x = (x - min(x) + EPS) / (max(x) - min(x) + EPS). \n"
                    f"Labels are assumed to have values between [0,1] "
                    f"and they are not normalised. "
                    f"This is to prevent accidental use of other encoding methods "
                    f"other than one-hot to represent multiple class labels.\n"
                    f"If the label values are intended to represent multiple labels, "
                    f"convert them to one hot / binary masks in multiple channels, "
                    f"with each channel representing one label only.\n"
                    f"Please read the dataset requirements section "
                    f"in docs/doc_data_loader.md for more detailed information."
                )
        # images should be 3D arrays
        for arr, name in zip(
            [moving_image, fixed_image], ["moving_image", "fixed_image"]
        ):
            if len(arr.shape) != 3 or min(arr.shape) <= 0:
                raise ValueError(
                    f"Sample {image_indices}'s {name}'s shape should be 3D"
                    f" and non-empty, got {arr.shape}."
                )
        # when data are labeled
        if moving_label is not None and fixed_label is not None:
            # labels should be 3D or 4D arrays
            for arr, name in zip(
                [moving_label, fixed_label], ["moving_label", "fixed_label"]
            ):
                if len(arr.shape) not in [3, 4]:
                    raise ValueError(
                        f"Sample {image_indices}'s {name}'s shape should be 3D or 4D. "
                        f"Got {arr.shape}."
                    )
            # image and label is better to have the same shape
            if moving_image.shape[:3] != moving_label.shape[:3]:  # pragma: no cover
                logger.warning(
                    f"Sample {image_indices}'s moving image and label "
                    f"have different shapes. "
                    f"moving_image.shape = {moving_image.shape}, "
                    f"moving_label.shape = {moving_label.shape}"
                )
            if fixed_image.shape[:3] != fixed_label.shape[:3]:  # pragma: no cover
                logger.warning(
                    f"Sample {image_indices}'s fixed image and label "
                    f"have different shapes. "
                    f"fixed_image.shape = {fixed_image.shape}, "
                    f"fixed_label.shape = {fixed_label.shape}"
                )
            # number of labels for fixed and fixed images should be the same
            num_labels_moving = (
                1 if len(moving_label.shape) == 3 else moving_label.shape[-1]
            )
            num_labels_fixed = (
                1 if len(fixed_label.shape) == 3 else fixed_label.shape[-1]
            )
            if num_labels_moving != num_labels_fixed:
                raise ValueError(
                    f"Sample {image_indices}'s moving image and fixed image "
                    f"have different numbers of labels. "
                    f"moving: {num_labels_moving}, fixed: {num_labels_fixed}"
                )

    def sample_image_label(
        self,
        moving_image: np.ndarray,
        fixed_image: np.ndarray,
        moving_label: Optional[np.ndarray],
        fixed_label: Optional[np.ndarray],
        image_indices: list,
    ):
        """
        Sample the image labels, only used in data_generator.

        :param moving_image:
        :param fixed_image:
        :param moving_label:
        :param fixed_label:
        :param image_indices:
        """
        self.validate_images_and_labels(
            moving_image, fixed_image, moving_label, fixed_label, image_indices
        )
        # unlabeled
        if moving_label is None or fixed_label is None:
            label_index = -1  # means no label
            indices = np.asarray(image_indices + [label_index], dtype=np.float32)
            yield dict(
                moving_image=moving_image, fixed_image=fixed_image, indices=indices
            )
        else:
            # labeled
            if len(moving_label.shape) == 4:  # multiple labels
                label_indices = get_label_indices(
                    moving_label.shape[3], self.sample_label  # type:ignore
                )
                for label_index in label_indices:
                    indices = np.asarray(
                        image_indices + [label_index], dtype=np.float32
                    )
                    yield dict(
                        moving_image=moving_image,
                        fixed_image=fixed_image,
                        indices=indices,
                        moving_label=moving_label[..., label_index],
                        fixed_label=fixed_label[..., label_index],
                    )
            else:  # only one label
                label_index = 0
                indices = np.asarray(image_indices + [label_index], dtype=np.float32)
                yield dict(
                    moving_image=moving_image,
                    fixed_image=fixed_image,
                    moving_label=moving_label,
                    fixed_label=fixed_label,
                    indices=indices,
                )


class FileLoader:
    """
    Interface / abstract class to load data from multiple directories.
    """

    def __init__(self, dir_paths: list, name: str, grouped: bool):
        """
        :param dir_paths: path to the directory of the data set
        :param name: name is used to identify the subdirectories or file names
        :param grouped: true if the data is grouped
        """
        assert isinstance(
            dir_paths, list
        ), f"dir_paths must be list of strings, got {dir_paths}"
        if len(set(dir_paths)) != len(dir_paths):
            raise ValueError(f"dir_paths have repeated elements: {dir_paths}")
        self.dir_paths = dir_paths
        self.name = name
        self.grouped = grouped
        # if grouped, group_struct[group_index] = list of data_index
        self.group_struct = None

    def set_data_structure(self):
        """
        Store the data structure in memory to retrieve data using data_index.
        """
        raise NotImplementedError

    def set_group_structure(self):
        """
        In addition to set_data_structure,
        store the group structure in the group_struct so that
        group_struct[group_index] = list of data_index
        and data can be retrieved data by
        data_index = group_struct[group_index][in_group_data_index]
        """
        raise NotImplementedError

    def get_data(self, index: Union[int, Tuple[int, ...]]) -> np.ndarray:
        """
        Get one data array by specifying an index.

        :param index: the data index which is required

          - for paired or unpaired, the index is one single int, data_index
          - for grouped, the index is a tuple of two ints,
            (group_index, in_group_data_index)

        :return: the data array at the specified index
        """
        raise NotImplementedError

    def get_data_ids(self) -> List:
        """
        Return the unique IDs of the data in this data set.
        This function is used to verify the consistency between
        moving and fixed images and label.
        """
        raise NotImplementedError

    def get_num_images(self) -> int:
        """
        Return the number of image in this data set.

        :return: int, number of images in this data set
        """
        raise NotImplementedError

    def get_num_groups(self) -> int:
        """
        Return the number of groups in grouped data set.

        :return: int, number of groups in this data set, if grouped
        """
        assert self.group_struct is not None
        return len(self.group_struct)

    def get_num_images_per_group(self) -> List[int]:
        """
        Return the number of images in each group.
        Each group must have at least one image.

        :return: a list of integers, representing the number of images in each group.
        """
        assert self.group_struct is not None
        num_images_per_group = [len(group) for group in self.group_struct]
        if min(num_images_per_group) == 0:
            group_ids = [
                len(group) for group_index, group in enumerate(self.group_struct)
            ]
            raise ValueError(f"Groups of ID {group_ids} are empty.")
        return num_images_per_group

    def close(self):
        """Close opened file handles if exist."""
        raise NotImplementedError
