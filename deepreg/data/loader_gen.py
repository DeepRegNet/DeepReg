import tensorflow as tf

import deepreg.data.preprocess as preprocess


class GeneratorDataLoader:
    def __init__(self):
        self.moving_image_shape = None
        self.fixed_image_shape = None
        self.num_indices = None
        self.num_images = None

    def get_generator(self):
        raise NotImplementedError

    def get_dataset(self, training, batch_size, repeat: bool, shuffle_buffer_size):
        dataset = tf.data.Dataset.from_generator(
            generator=self.get_generator,
            output_types=((tf.float32, tf.float32, tf.float32, tf.float32), tf.float32),
            output_shapes=((self.moving_image_shape, self.fixed_image_shape, self.moving_image_shape, self.num_indices),
                           self.fixed_image_shape),
        )
        dataset = preprocess.preprocess(dataset=dataset,
                                        moving_image_shape=self.moving_image_shape,
                                        fixed_image_shape=self.fixed_image_shape,
                                        training=training,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat,
                                        batch_size=batch_size)
        return dataset
