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
        transformed = layer_util.resample_linear(source=image,
                                                 sample_coords=layer_util.warp_grid(grid_ref, transforms))
        return transformed

    @tf.function
    def transform(self, inputs, labels):
        """
        :param inputs: (moving_image, fixed_image, moving_label)
                    moving_image, shape = [batch, m_dim1, m_dim2, m_dim3]
                    fixed_image, shape = [batch, f_dim1, f_dim2, f_dim3]
                    moving_label, shape = [batch, m_dim1, m_dim2, m_dim3]
        :param labels: fixed_label, shape = [batch, f_dim1, f_dim2, f_dim3]
        :param indices: a 2 element array, [sample_index, label_index]
        :return:
        """

        moving_image, fixed_image, moving_label, indices = inputs
        fixed_label = labels

        moving_transforms = self._gen_transforms()
        fixed_transforms = self._gen_transforms()

        moving_image = self._transform(moving_image, self._moving_grid_ref, moving_transforms)
        moving_label = self._transform(moving_label, self._moving_grid_ref, moving_transforms)
        fixed_image = self._transform(fixed_image, self._fixed_grid_ref, fixed_transforms)
        fixed_label = self._transform(fixed_label, self._fixed_grid_ref, fixed_transforms)

        return (moving_image, fixed_image, moving_label, indices), fixed_label
