import tensorflow as tf

from deepreg.data import augmentation as aug


class DataLoader:
    def __init__(self):
        """
        The following four attributed need to be defined in the data loader where
        - moving_image_shape is the shape of moving images fed into dataset, it is assumed to be 3d, [dim1, dim2, dim3]
        - fixed_image_shape is the shape of fixed images fed into dataset, it is assumed to be 3d, [dim1, dim2, dim3]
        - num_indices is the number of indices fed into dataset, the indices are used to identify the ID of data sample
        - num_images is the number of the samples (image pairs) in the entire dataset
        """
        self.moving_image_shape = None
        self.fixed_image_shape = None
        self.num_indices = None
        self.num_images = None

    def get_dataset(self):
        raise NotImplementedError

    def get_dataset_and_preprocess(self, training, batch_size, repeat: bool, shuffle_buffer_size):
        dataset = preprocess(dataset=self.get_dataset(),
                             moving_image_shape=self.moving_image_shape,
                             fixed_image_shape=self.fixed_image_shape,
                             training=training,
                             shuffle_buffer_size=shuffle_buffer_size,
                             repeat=repeat,
                             batch_size=batch_size)
        return dataset

    def split_indices(self, indices: list):
        raise NotImplementedError

    def image_index_to_dir(self, image_index):
        raise NotImplementedError


class GeneratorDataLoader(DataLoader):
    def __init__(self):
        super(GeneratorDataLoader, self).__init__()

    def get_generator(self):
        raise NotImplementedError

    def get_dataset(self):
        return tf.data.Dataset.from_generator(
            generator=self.get_generator,
            output_types=((tf.float32, tf.float32, tf.float32, tf.float32), tf.float32),
            output_shapes=((self.moving_image_shape, self.fixed_image_shape, self.moving_image_shape, self.num_indices),
                           self.fixed_image_shape),
        )

    def split_indices(self, indices: list):
        raise NotImplementedError

    def image_index_to_dir(self, image_index):
        raise NotImplementedError


def preprocess(dataset,
               moving_image_shape, fixed_image_shape,
               training,
               shuffle_buffer_size, repeat: bool, batch_size):
    """
    shuffle, repeat, batch, augmentation
    :param dataset:
    :param moving_image_shape:
    :param fixed_image_shape:
    :param training:
    :param batch_size:
    :param repeat:
    :param shuffle_buffer_size:
    :return:
    """
    if training and shuffle_buffer_size > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.batch(batch_size, drop_remainder=training)
    if training:
        # TODO add cropping, but crop first or rotation first?
        affine_transform = aug.AffineTransformation3D(moving_image_size=moving_image_shape,
                                                      fixed_image_size=fixed_image_shape,
                                                      batch_size=batch_size)
        dataset = dataset.map(affine_transform.transform)
    return dataset
