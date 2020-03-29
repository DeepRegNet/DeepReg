import random

import tensorflow as tf

import deepreg.data.preprocess as preprocess


class BasicDataLoader:
    def __init__(self):
        self.moving_image_shape = None
        self.fixed_image_shape = None
        self.sample_label = None
        self.num_images = None

    def get_label_indices(self, num_labels):
        if self.sample_label == "sample":  # sample a random label
            return [random.randrange(num_labels)]
        elif self.sample_label == "first":  # use the first label
            return [0]
        elif self.sample_label == "all":  # use all labels
            return list(range(num_labels))
        else:
            raise ValueError("Unknown label sampling policy %s" % self.sample_label)

    def get_generator(self):
        raise NotImplementedError

    def _get_dataset(self):
        return tf.data.Dataset.from_generator(
            generator=self.get_generator,
            output_types=((tf.float32, tf.float32, tf.float32, tf.float32), tf.float32),
            output_shapes=((self.moving_image_shape, self.fixed_image_shape, self.moving_image_shape, 2),
                           self.fixed_image_shape),
        )

    def get_dataset(self, training, batch_size, repeat: bool, shuffle_buffer_size):
        dataset = self._get_dataset()
        dataset = preprocess.preprocess(dataset=dataset,
                                        moving_image_shape=self.moving_image_shape,
                                        fixed_image_shape=self.fixed_image_shape,
                                        training=training,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat,
                                        batch_size=batch_size)
        return dataset
