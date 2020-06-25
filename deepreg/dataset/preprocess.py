import tensorflow as tf

import deepreg.model.layer_util as layer_util


class AffineTransformation3D:
    def __init__(self, moving_image_size, fixed_image_size, batch_size, scale=0.1):
        self._batch_size = batch_size
        self._scale = scale
        self._moving_grid_ref = layer_util.get_reference_grid(grid_size=moving_image_size)
        self._fixed_grid_ref = layer_util.get_reference_grid(grid_size=fixed_image_size)

    def _gen_transforms(self):
        return layer_util.random_transform_generator(batch_size=self._batch_size, scale=self._scale)

    @staticmethod
    def _transform(image, grid_ref, transforms):
        """

        :param image: shape = [batch, dim1, dim2, dim3]
        :param grid_ref: shape = [dim1, dim2, dim3, 3]
        :param transforms: shape = [batch, 4, 3]
        :return: shape = [batch, dim1, dim2, dim3]
        """
        transformed = layer_util.resample(vol=image,
                                          loc=layer_util.warp_grid(grid_ref, transforms))
        return transformed

    @tf.function
    def transform(self, inputs: dict):
        """
        :param inputs:
                    moving_image, shape = (batch, m_dim1, m_dim2, m_dim3)
                    fixed_image, shape = (batch, f_dim1, f_dim2, f_dim3)
                    moving_label, shape = (batch, m_dim1, m_dim2, m_dim3)
                    fixed_label, shape = (batch, f_dim1, f_dim2, f_dim3)
                    indices, shape = (num_indices, )
        :return:
        """

        moving_image = inputs.get("moving_image")
        fixed_image = inputs.get("fixed_image")
        moving_label = inputs.get("moving_label", None)
        fixed_label = inputs.get("fixed_label", None)
        indices = inputs.get("indices")

        moving_transforms = self._gen_transforms()
        fixed_transforms = self._gen_transforms()

        moving_image = self._transform(moving_image, self._moving_grid_ref, moving_transforms)
        fixed_image = self._transform(fixed_image, self._fixed_grid_ref, fixed_transforms)

        if moving_label is None:  # unlabeled
            return dict(
                moving_image=moving_image,
                fixed_image=fixed_image,
                indices=indices,
            )
        
        moving_label = self._transform(moving_label, self._moving_grid_ref, moving_transforms)
        fixed_label = self._transform(fixed_label, self._fixed_grid_ref, fixed_transforms)

        return dict(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_label=moving_label,
            fixed_label=fixed_label,
            indices=indices,
        )


def preprocess(dataset,
               moving_image_shape, fixed_image_shape,
               training,
               shuffle_buffer_num_batch, repeat: bool, batch_size):
    """
    shuffle, repeat, batch, augmentation
    :param dataset:
    :param moving_image_shape:
    :param fixed_image_shape:
    :param training:
    :param batch_size:
    :param repeat:
    :param shuffle_buffer_num_batch:
    :return:
    """
    # shuffle / repeat / batch / preprocess
    if training and shuffle_buffer_num_batch > 0:
        dataset = dataset.shuffle(buffer_size=batch_size * shuffle_buffer_num_batch)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size=batch_size,
                            drop_remainder=training)
    if training:
        # TODO add cropping, but crop first or rotation first?
        affine_transform = AffineTransformation3D(moving_image_size=moving_image_shape,
                                                  fixed_image_size=fixed_image_shape,
                                                  batch_size=batch_size)
        dataset = dataset.map(affine_transform.transform)
    return dataset
