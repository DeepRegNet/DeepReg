import tensorflow as tf

import src.data.augmentation as aug


class BasicDataLoader:
    def __init__(self):
        self.moving_image_shape = None
        self.fixed_image_shape = None

    def get_generator(self):
        raise NotImplementedError

    def _get_dataset(self):
        return tf.data.Dataset.from_generator(
            generator=self.get_generator,
            output_types=((tf.float32, tf.float32, tf.float32), tf.float32, tf.float32),
            output_shapes=((self.moving_image_shape, self.fixed_image_shape,
                            self.moving_image_shape), self.fixed_image_shape,
                           2,),
        )

    def get_dataset(self, training, batch_size, shuffle_buffer_size):
        dataset = self._get_dataset()
        if training and shuffle_buffer_size > 0:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=training)
        if training:
            # TODO add cropping, but crop first or rotation first?
            affine_transform = aug.AffineTransformation3D(moving_image_size=self.moving_image_shape,
                                                          fixed_image_size=self.fixed_image_shape,
                                                          batch_size=batch_size)
            dataset = dataset.map(affine_transform.transform)
        return dataset
