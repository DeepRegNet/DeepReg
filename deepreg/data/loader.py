import tensorflow as tf

from deepreg.data.preprocess import preprocess
from deepreg.data.tfrecord import load_tfrecords, get_tfrecords_filenames


class DataLoader:
    def __init__(self):
        """
        The following four attributed need to be defined in the data loader where
        - moving_image_shape is the shape of moving images fed into dataset, it is assumed to be 3d, [dim1, dim2, dim3]
        - fixed_image_shape is the shape of fixed images fed into dataset, it is assumed to be 3d, [dim1, dim2, dim3]
        - num_indices is the number of indices fed into dataset, the indices are used to identify the ID of data sample
          default is 2, for image_index and label_index
        - num_images is the number of the samples (image pairs) in the entire dataset
        """
        self.moving_image_shape = None
        self.fixed_image_shape = None
        self.num_indices = 2
        self.num_images = None

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


class GeneratorDataLoader(DataLoader):
    def __init__(self):
        super(GeneratorDataLoader, self).__init__()
        self.tfrecord_dir = None

    def get_generator(self):
        raise NotImplementedError

    def get_dataset(self):
        if self.tfrecord_dir != "" and self.tfrecord_dir is not None:
            return load_tfrecords(filenames=get_tfrecords_filenames(tfrecord_dir=self.tfrecord_dir),
                                  moving_image_shape=self.moving_image_shape,
                                  fixed_image_shape=self.fixed_image_shape,
                                  num_indices=self.num_indices)
        else:
            return tf.data.Dataset.from_generator(
                generator=self.get_generator,
                output_types=((tf.float32, tf.float32, tf.float32, tf.float32), tf.float32),
                output_shapes=(
                    (self.moving_image_shape, self.fixed_image_shape, self.moving_image_shape, self.num_indices),
                    self.fixed_image_shape),
            )
