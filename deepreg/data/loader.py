from abc import ABC

import numpy as np
import tensorflow as tf

from deepreg.data.preprocess import preprocess
from deepreg.data.util import get_label_indices


class DataLoader:
    def __init__(self, num_indices, sample_label, seed=None):
        self.num_indices = num_indices  # number of indices to identify a sample
        self.sample_label = sample_label
        self.seed = seed  # used for sampling

    @property
    def moving_image_shape(self) -> tuple:
        raise NotImplementedError

    @property
    def fixed_image_shape(self) -> tuple:
        raise NotImplementedError

    @property
    def num_samples(self) -> int:
        """
        Return the number of samples in the dataset for one epoch
        :return:
        """
        raise NotImplementedError

    def get_dataset(self):
        raise NotImplementedError

    def get_dataset_and_preprocess(self, training, batch_size, repeat: bool, shuffle_buffer_num_batch):
        dataset = preprocess(dataset=self.get_dataset(),
                             moving_image_shape=self.moving_image_shape,
                             fixed_image_shape=self.fixed_image_shape,
                             training=training,
                             shuffle_buffer_num_batch=shuffle_buffer_num_batch,
                             repeat=repeat,
                             batch_size=batch_size)
        return dataset

    def validate_images_and_labels(self, moving_image: np.ndarray, fixed_image: np.ndarray,
                                   moving_label: np.ndarray, fixed_label: np.ndarray, image_indices: list):
        for arr, name in zip([moving_image, fixed_image, moving_label, fixed_label],
                             ["moving_image", "fixed_image", "moving_label", "fixed_label"]):
            if np.min(arr) < 0 or np.max(arr) > 1:
                raise ValueError("Sample {}'s {} has value outside of [0,1]."
                                 "Images are assumed to be between [0, 255] "
                                 "and labels are assumed to be between [0, 1]".format(image_indices, name))
        for arr, name in zip([moving_image, moving_label],
                             ["moving_image", "moving_label"]):
            if arr.shape[:3] != self.moving_image_shape:
                raise ValueError("Sample {}'s {} has different shape (width, height, depth) from required."
                                 "Expected {} but got {}.".format(image_indices, name, self.moving_image_shape,
                                                                  arr.shape[:3]))
        for arr, name in zip([fixed_image, fixed_label],
                             ["fixed_image", "fixed_label"]):
            if arr.shape[:3] != self.fixed_image_shape:
                raise ValueError("Sample {}'s {} has different shape (width, height, depth) from required."
                                 "Expected {} but got {}.".format(image_indices, name, self.fixed_image_shape,
                                                                  arr.shape[:3]))
        num_labels_moving = 1 if len(moving_label.shape) == 3 else moving_label.shape[-1]
        num_labels_fixed = 1 if len(fixed_label.shape) == 3 else fixed_label.shape[-1]
        if num_labels_moving != num_labels_fixed:
            raise ValueError(
                "Sample {}'s moving image and fixed image have different numbers of labels."
                "moving: {}, fixed: {}".format(image_indices, num_labels_moving, num_labels_fixed))

    def sample_image_label(self, moving_image: np.ndarray, fixed_image: np.ndarray,
                           moving_label: np.ndarray, fixed_label: np.ndarray, image_indices: list):
        self.validate_images_and_labels(moving_image, fixed_image, moving_label, fixed_label, image_indices)
        if len(moving_label.shape) == 4:  # multiple labels
            label_indices = get_label_indices(moving_label.shape[3], self.sample_label)
            for label_index in label_indices:
                indices = np.asarray(image_indices + [label_index], dtype=np.float32)
                inputs = (moving_image, fixed_image, moving_label[..., label_index], indices)
                labels = fixed_label[..., label_index]
                yield inputs, labels
        else:  # only one label
            label_index = 0
            indices = np.asarray(image_indices + [label_index], dtype=np.float32)
            inputs = (moving_image, fixed_image, moving_label, indices)
            labels = fixed_label
            yield inputs, labels


class PairedDataLoader(DataLoader, ABC):
    def __init__(self, moving_image_shape: (list, tuple), fixed_image_shape: (list, tuple), **kwargs):
        """
        num_indices = 2 corresponding to (image_index, label_index)
        :param moving_image_shape: (width, height, depth)
        :param fixed_image_shape:  (width, height, depth)
        """
        super(PairedDataLoader, self).__init__(num_indices=2, **kwargs)
        if len(moving_image_shape) != 3 or len(fixed_image_shape) != 3:
            raise ValueError("moving_image_shape and fixed_image_shape have to be length of three,"
                             "corresponding to (width, height, depth)")
        self._moving_image_shape = tuple(moving_image_shape)
        self._fixed_image_shape = tuple(fixed_image_shape)

    @property
    def moving_image_shape(self) -> tuple:
        return self._moving_image_shape

    @property
    def fixed_image_shape(self) -> tuple:
        return self._fixed_image_shape


class UnpairedDataLoader(DataLoader, ABC):
    def __init__(self, image_shape: (list, tuple), **kwargs):
        """
        - image_shape is the shape of images fed into dataset, it is assumed to be 3d, [dim1, dim2, dim3]
          moving_image_shape = fixed_image_shape = image_shape
        """
        super(UnpairedDataLoader, self).__init__(num_indices=3, **kwargs)
        if len(image_shape) != 3:
            raise ValueError("image_shape has to be length of three,"
                             "corresponding to (width, height, depth)")
        self.image_shape = tuple(image_shape)

    @property
    def moving_image_shape(self) -> tuple:
        return self.image_shape

    @property
    def fixed_image_shape(self) -> tuple:
        return self.image_shape


class GeneratorDataLoader(DataLoader, ABC):
    def __init__(self, **kwargs):
        super(GeneratorDataLoader, self).__init__(**kwargs)

    def get_generator(self):
        raise NotImplementedError

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            generator=self.get_generator,
            output_types=((tf.float32, tf.float32, tf.float32, tf.float32), tf.float32),
            output_shapes=(
                (self.moving_image_shape, self.fixed_image_shape, self.moving_image_shape, self.num_indices),
                self.fixed_image_shape),
        )
