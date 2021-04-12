"""
Module containing data augmentation techniques.
  - 3D Affine/DDF Transforms for moving and fixed images.
"""

from abc import abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from deepreg.model.layer import Resize3d
from deepreg.model.layer_util import get_reference_grid, resample, warp_grid
from deepreg.registry import REGISTRY


class RandomTransformation3D(tf.keras.layers.Layer):
    """
    An interface for different types of transformation.
    """

    def __init__(
        self,
        moving_image_size: Tuple[int, ...],
        fixed_image_size: Tuple[int, ...],
        batch_size: int,
        name: str = "RandomTransformation3D",
        trainable: bool = False,
    ):
        """
        Abstract class for image transformation.

        :param moving_image_size: (m_dim1, m_dim2, m_dim3)
        :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
        :param batch_size: total number of samples consumed per step, over all devices.
        :param name: name of layer
        :param trainable: if this layer is trainable
        """
        super().__init__(trainable=trainable, name=name)
        self.moving_image_size = moving_image_size
        self.fixed_image_size = fixed_image_size
        self.batch_size = batch_size
        self.moving_grid_ref = get_reference_grid(grid_size=moving_image_size)
        self.fixed_grid_ref = get_reference_grid(grid_size=fixed_image_size)

    @abstractmethod
    def gen_transform_params(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generates transformation parameters for moving and fixed image.

        :return: two tensors
        """

    @staticmethod
    @abstractmethod
    def transform(
        image: tf.Tensor, grid_ref: tf.Tensor, params: tf.Tensor
    ) -> tf.Tensor:
        """
        Transforms the reference grid and then resample the image.

        :param image: shape = (batch, dim1, dim2, dim3)
        :param grid_ref: shape = (dim1, dim2, dim3, 3)
        :param params: parameters for transformation
        :return: shape = (batch, dim1, dim2, dim3)
        """

    def call(self, inputs: Dict[str, tf.Tensor], **kwargs) -> Dict[str, tf.Tensor]:
        """
        Creates random params for the input images and their labels,
        and params them based on the resampled reference grids.
        :param inputs: a dict having multiple tensors
            if labeled:
                moving_image, shape = (batch, m_dim1, m_dim2, m_dim3)
                fixed_image, shape = (batch, f_dim1, f_dim2, f_dim3)
                moving_label, shape = (batch, m_dim1, m_dim2, m_dim3)
                fixed_label, shape = (batch, f_dim1, f_dim2, f_dim3)
                indices, shape = (batch, num_indices)
            else, unlabeled:
                moving_image, shape = (batch, m_dim1, m_dim2, m_dim3)
                fixed_image, shape = (batch, f_dim1, f_dim2, f_dim3)
                indices, shape = (batch, num_indices)
        :param kwargs: other arguments
        :return: dictionary with the same structure as inputs
        """

        moving_image = inputs["moving_image"]
        fixed_image = inputs["fixed_image"]
        indices = inputs["indices"]

        moving_params, fixed_params = self.gen_transform_params()

        moving_image = self.transform(moving_image, self.moving_grid_ref, moving_params)
        fixed_image = self.transform(fixed_image, self.fixed_grid_ref, fixed_params)

        if "moving_label" not in inputs:  # unlabeled
            return dict(
                moving_image=moving_image, fixed_image=fixed_image, indices=indices
            )
        moving_label = inputs["moving_label"]
        fixed_label = inputs["fixed_label"]

        moving_label = self.transform(moving_label, self.moving_grid_ref, moving_params)
        fixed_label = self.transform(fixed_label, self.fixed_grid_ref, fixed_params)

        return dict(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_label=moving_label,
            fixed_label=fixed_label,
            indices=indices,
        )

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["moving_image_size"] = self.moving_image_size
        config["fixed_image_size"] = self.fixed_image_size
        config["batch_size"] = self.batch_size
        return config


@REGISTRY.register_data_augmentation(name="affine")
class RandomAffineTransform3D(RandomTransformation3D):
    """Apply random affine transformation to moving/fixed images separately."""

    def __init__(
        self,
        moving_image_size: Tuple[int, ...],
        fixed_image_size: Tuple[int, ...],
        batch_size: int,
        scale: float = 0.1,
        name: str = "RandomAffineTransform3D",
        **kwargs,
    ):
        """
        Init.

        :param moving_image_size: (m_dim1, m_dim2, m_dim3)
        :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
        :param batch_size: total number of samples consumed per step, over all devices.
        :param scale: a positive float controlling the scale of transformation
        :param name: name of the layer
        :param kwargs: additional arguments
        """
        super().__init__(
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            batch_size=batch_size,
            name=name,
            **kwargs,
        )
        self.scale = scale

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["scale"] = self.scale
        return config

    def gen_transform_params(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Function that generates the random 3D transformation parameters
        for a batch of data for moving and fixed image.

        :return: a tuple of tensors, each has shape = (batch, 4, 3)
        """
        theta = gen_rand_affine_transform(
            batch_size=self.batch_size * 2, scale=self.scale
        )
        return theta[: self.batch_size], theta[self.batch_size :]

    @staticmethod
    def transform(
        image: tf.Tensor, grid_ref: tf.Tensor, params: tf.Tensor
    ) -> tf.Tensor:
        """
        Transforms the reference grid and then resample the image.

        :param image: shape = (batch, dim1, dim2, dim3)
        :param grid_ref: shape = (dim1, dim2, dim3, 3)
        :param params: shape = (batch, 4, 3)
        :return: shape = (batch, dim1, dim2, dim3)
        """
        return resample(vol=image, loc=warp_grid(grid_ref, params))


@REGISTRY.register_data_augmentation(name="ddf")
class RandomDDFTransform3D(RandomTransformation3D):
    """Apply random DDF transformation to moving/fixed images separately."""

    def __init__(
        self,
        moving_image_size: Tuple[int, ...],
        fixed_image_size: Tuple[int, ...],
        batch_size: int,
        field_strength: int = 1,
        low_res_size: tuple = (1, 1, 1),
        name: str = "RandomDDFTransform3D",
        **kwargs,
    ):
        """
        Creates a DDF transformation for data augmentation.

        To simulate smooth deformation fields, we interpolate from a low resolution
        field of size low_res_size using linear interpolation. The variance of the
        deformation field is drawn from a uniform variable
        between [0, field_strength].

        :param moving_image_size: tuple
        :param fixed_image_size: tuple
        :param batch_size: total number of samples consumed per step, over all devices.
        :param field_strength: int = 1. It is used as the upper bound for the
        deformation field variance
        :param low_res_size: tuple = (1, 1, 1).
        :param name: name of layer
        :param kwargs: additional arguments
        """

        super().__init__(
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            batch_size=batch_size,
            name=name,
            **kwargs,
        )

        assert tuple(low_res_size) <= tuple(moving_image_size)
        assert tuple(low_res_size) <= tuple(fixed_image_size)

        self.field_strength = field_strength
        self.low_res_size = low_res_size

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["field_strength"] = self.field_strength
        config["low_res_size"] = self.low_res_size
        return config

    def gen_transform_params(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Generates two random ddf fields for moving and fixed images.

        :return: tuple, one has shape = (batch, m_dim1, m_dim2, m_dim3, 3)
            another one has shape = (batch, f_dim1, f_dim2, f_dim3, 3)
        """
        moving = gen_rand_ddf(
            image_size=self.moving_image_size,
            batch_size=self.batch_size,
            field_strength=self.field_strength,
            low_res_size=self.low_res_size,
        )
        fixed = gen_rand_ddf(
            image_size=self.fixed_image_size,
            batch_size=self.batch_size,
            field_strength=self.field_strength,
            low_res_size=self.low_res_size,
        )
        return moving, fixed

    @staticmethod
    def transform(
        image: tf.Tensor, grid_ref: tf.Tensor, params: tf.Tensor
    ) -> tf.Tensor:
        """
        Transforms the reference grid and then resample the image.

        :param image: shape = (batch, dim1, dim2, dim3)
        :param grid_ref: shape = (dim1, dim2, dim3, 3)
        :param params: DDF, shape = (batch, dim1, dim2, dim3, 3)
        :return: shape = (batch, dim1, dim2, dim3)
        """
        return resample(vol=image, loc=grid_ref[None, ...] + params)


def resize_inputs(
    inputs: Dict[str, tf.Tensor],
    moving_image_size: Tuple[int, ...],
    fixed_image_size: tuple,
) -> Dict[str, tf.Tensor]:
    """
    Resize inputs
    :param inputs:
        if labeled:
            moving_image, shape = (None, None, None)
            fixed_image, shape = (None, None, None)
            moving_label, shape = (None, None, None)
            fixed_label, shape = (None, None, None)
            indices, shape = (num_indices, )
        else, unlabeled:
            moving_image, shape = (None, None, None)
            fixed_image, shape = (None, None, None)
            indices, shape = (num_indices, )
    :param moving_image_size: Tuple[int, ...], (m_dim1, m_dim2, m_dim3)
    :param fixed_image_size: Tuple[int, ...], (f_dim1, f_dim2, f_dim3)
    :return:
        if labeled:
            moving_image, shape = (m_dim1, m_dim2, m_dim3)
            fixed_image, shape = (f_dim1, f_dim2, f_dim3)
            moving_label, shape = (m_dim1, m_dim2, m_dim3)
            fixed_label, shape = (f_dim1, f_dim2, f_dim3)
            indices, shape = (num_indices, )
        else, unlabeled:
            moving_image, shape = (m_dim1, m_dim2, m_dim3)
            fixed_image, shape = (f_dim1, f_dim2, f_dim3)
            indices, shape = (num_indices, )
    """
    moving_image = inputs["moving_image"]
    fixed_image = inputs["fixed_image"]
    indices = inputs["indices"]

    moving_resize_layer = Resize3d(shape=moving_image_size)
    fixed_resize_layer = Resize3d(shape=fixed_image_size)

    moving_image = moving_resize_layer(moving_image)
    fixed_image = fixed_resize_layer(fixed_image)

    if "moving_label" not in inputs:  # unlabeled
        return dict(moving_image=moving_image, fixed_image=fixed_image, indices=indices)
    moving_label = inputs["moving_label"]
    fixed_label = inputs["fixed_label"]

    moving_label = moving_resize_layer(moving_label)
    fixed_label = fixed_resize_layer(fixed_label)

    return dict(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_label=moving_label,
        fixed_label=fixed_label,
        indices=indices,
    )


def gen_rand_affine_transform(
    batch_size: int, scale: float, seed: Optional[int] = None
) -> tf.Tensor:
    """
    Function that generates a random 3D transformation parameters for a batch of data.

    for 3D coordinates, affine transformation is

    .. code-block:: text

        [[x' y' z' 1]] = [[x y z 1]] * [[* * * 0]
                                        [* * * 0]
                                        [* * * 0]
                                        [* * * 1]]

    where each * represents a degree of freedom,
    so there are in total 12 degrees of freedom
    the equation can be denoted as

        new = old * T

    where

    - new is the transformed coordinates, of shape (1, 4)
    - old is the original coordinates, of shape (1, 4)
    - T is the transformation matrix, of shape (4, 4)

    the equation can be simplified to

    .. code-block:: text

        [[x' y' z']] = [[x y z 1]] * [[* * *]
                                      [* * *]
                                      [* * *]
                                      [* * *]]

    so that

        new = old * T

    where

    - new is the transformed coordinates, of shape (1, 3)
    - old is the original coordinates, of shape (1, 4)
    - T is the transformation matrix, of shape (4, 3)

    Given original and transformed coordinates,
    we can calculate the transformation matrix using

        x = np.linalg.lstsq(a, b)

    such that

        a x = b

    In our case,

    - a = old
    - b = new
    - x = T

    To generate random transformation,
    we choose to add random perturbation to corner coordinates as follows:
    for corner of coordinates (x, y, z), the noise is

        -(x, y, z) .* (r1, r2, r3)

    where ri is a random number between (0, scale).
    So

        (x', y', z') = (x, y, z) .* (1-r1, 1-r2, 1-r3)

    Thus, we can directly sample between 1-scale and 1 instead

    We choose to calculate the transformation based on
    four corners in a cube centered at (0, 0, 0).
    A cube is shown as below, where

    - C = (-1, -1, -1)
    - G = (-1, -1, 1)
    - D = (-1, 1, -1)
    - A = (1, -1, -1)

    .. code-block:: text

                    G — — — — — — — — H
                  / |               / |
                /   |             /   |
              /     |           /     |
            /       |         /       |
          /         |       /         |
        E — — — — — — — — F           |
        |           |     |           |
        |           |     |           |
        |           C — — | — — — — — D
        |         /       |         /
        |       /         |       /
        |     /           |     /
        |   /             |   /
        | /               | /
        A — — — — — — — — B

    :param batch_size: total number of samples consumed per step, over all devices.
    :param scale: a float number between 0 and 1
    :param seed: control the randomness
    :return: shape = (batch, 4, 3)
    """

    assert 0 <= scale <= 1
    np.random.seed(seed)
    noise = np.random.uniform(1 - scale, 1, [batch_size, 4, 3])  # shape = (batch, 4, 3)

    # old represents four corners of a cube
    # corresponding to the corner C G D A as shown above
    old = np.tile(
        [[[-1, -1, -1, 1], [-1, -1, 1, 1], [-1, 1, -1, 1], [1, -1, -1, 1]]],
        [batch_size, 1, 1],
    )  # shape = (batch, 4, 4)
    new = old[:, :, :3] * noise  # shape = (batch, 4, 3)

    theta = np.array(
        [np.linalg.lstsq(old[k], new[k], rcond=-1)[0] for k in range(batch_size)]
    )  # shape = (batch, 4, 3)

    return tf.cast(theta, dtype=tf.float32)


def gen_rand_ddf(
    batch_size: int,
    image_size: Tuple[int, ...],
    field_strength: Union[Tuple, List, int, float],
    low_res_size: Union[Tuple, List],
    seed: Optional[int] = None,
) -> tf.Tensor:
    """
    Function that generates a random 3D DDF for a batch of data.

    :param batch_size: total number of samples consumed per step, over all devices.
    :param image_size:
    :param field_strength: maximum field strength, computed as a U[0,field_strength]
    :param low_res_size: low_resolution deformation field that will be upsampled to
        the original size in order to get smooth and more realistic fields.
    :param seed: control the randomness
    :return:
    """

    np.random.seed(seed)
    low_res_strength = np.random.uniform(0, field_strength, (batch_size, 1, 1, 1, 3))
    low_res_field = low_res_strength * np.random.randn(
        batch_size, low_res_size[0], low_res_size[1], low_res_size[2], 3
    )
    high_res_field = Resize3d(shape=image_size)(low_res_field)
    return high_res_field
