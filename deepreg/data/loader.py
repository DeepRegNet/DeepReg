from abc import ABC

import tensorflow as tf

from deepreg.data.preprocess import preprocess


class DataLoader:
    def __init__(self, num_indices):
        self.num_indices = num_indices  # number of indices to identify a sample

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


class PairedDataLoader(DataLoader, ABC):
    def __init__(self, moving_image_shape: (list, tuple), fixed_image_shape: (list, tuple)):
        """
        num_indices = 2 corresponding to (image_index, label_index)
        :param moving_image_shape: (width, height, depth)
        :param fixed_image_shape:  (width, height, depth)
        """
        super(PairedDataLoader, self).__init__(num_indices=2)
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
    def __init__(self, image_shape: (list, tuple)):
        """
        - image_shape is the shape of images fed into dataset, it is assumed to be 3d, [dim1, dim2, dim3]
          moving_image_shape = fixed_image_shape = image_shape
        """
        super(UnpairedDataLoader, self).__init__(num_indices=3)
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
