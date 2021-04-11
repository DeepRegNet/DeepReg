"""Provide regularization functions and classes for ddf."""
from typing import Callable

import tensorflow as tf

from deepreg.registry import REGISTRY


def gradient_dx(fx: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on x-axis of a 3D tensor using central finite difference.

    It moves the tensor along axis 1 to calculate the approximate gradient, the x axis,
    dx[i] = (x[i+1] - x[i-1]) / 2.

    :param fx: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fx[:, 2:, 1:-1, 1:-1] - fx[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(fy: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on y-axis of a 3D tensor using central finite difference.

    It moves the tensor along axis 2 to calculate the approximate gradient, the y axis,
    dy[i] = (y[i+1] - y[i-1]) / 2.

    :param fy: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fy[:, 1:-1, 2:, 1:-1] - fy[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(fz: tf.Tensor) -> tf.Tensor:
    """
    Calculate gradients on z-axis of a 3D tensor using central finite difference.

    It moves the tensor along axis 3 to calculate the approximate gradient, the z axis,
    dz[i] = (z[i+1] - z[i-1]) / 2.

    :param fz: shape = (batch, m_dim1, m_dim2, m_dim3)
    :return: shape = (batch, m_dim1-2, m_dim2-2, m_dim3-2)
    """
    return (fz[:, 1:-1, 1:-1, 2:] - fz[:, 1:-1, 1:-1, :-2]) / 2


def gradient_dxyz(fxyz: tf.Tensor, fn: Callable) -> tf.Tensor:
    """
    Calculate gradients on x,y,z-axis of a tensor using central finite difference.

    The gradients are calculated along x, y, z separately then stacked together.

    :param fxyz: shape = (..., 3)
    :param fn: function to call
    :return: shape = (..., 3)
    """
    return tf.stack([fn(fxyz[..., i]) for i in [0, 1, 2]], axis=4)


@REGISTRY.register_loss(name="gradient")
class GradientNorm(tf.keras.layers.Layer):
    """
    Calculate the L1/L2 norm of ddf using central finite difference.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, l1: bool = False, name: str = "GradientNorm", **kwargs):
        """
        Init.

        :param l1: bool true if calculate L1 norm, otherwise L2 norm
        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)
        self.l1 = l1

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        dfdx = gradient_dxyz(ddf, gradient_dx)
        dfdy = gradient_dxyz(ddf, gradient_dy)
        dfdz = gradient_dxyz(ddf, gradient_dz)
        if self.l1:
            norms = tf.abs(dfdx) + tf.abs(dfdy) + tf.abs(dfdz)
        else:
            norms = dfdx ** 2 + dfdy ** 2 + dfdz ** 2
        return tf.reduce_mean(norms, axis=[1, 2, 3, 4])

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["l1"] = self.l1
        return config


@REGISTRY.register_loss(name="bending")
class BendingEnergy(tf.keras.layers.Layer):
    """
    Calculate the bending energy of ddf using central finite difference.

    y_true and y_pred have to be at least 5d tensor, including batch axis.
    """

    def __init__(self, name: str = "BendingEnergy", **kwargs):
        """
        Init.

        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name)

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        """
        Return a scalar loss.

        :param inputs: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
        :param kwargs: additional arguments.
        :return: shape = (batch, )
        """
        assert len(inputs.shape) == 5
        ddf = inputs
        # first order gradient
        # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
        dfdx = gradient_dxyz(ddf, gradient_dx)
        dfdy = gradient_dxyz(ddf, gradient_dy)
        dfdz = gradient_dxyz(ddf, gradient_dz)

        # second order gradient
        # (batch, m_dim1-4, m_dim2-4, m_dim3-4, 3)
        dfdxx = gradient_dxyz(dfdx, gradient_dx)
        dfdyy = gradient_dxyz(dfdy, gradient_dy)
        dfdzz = gradient_dxyz(dfdz, gradient_dz)
        dfdxy = gradient_dxyz(dfdx, gradient_dy)
        dfdyz = gradient_dxyz(dfdy, gradient_dz)
        dfdxz = gradient_dxyz(dfdx, gradient_dz)

        # (dx + dy + dz) ** 2 = dxx + dyy + dzz + 2*(dxy + dyz + dzx)
        energy = dfdxx ** 2 + dfdyy ** 2 + dfdzz ** 2
        energy += 2 * dfdxy ** 2 + 2 * dfdxz ** 2 + 2 * dfdyz ** 2
        return tf.reduce_mean(energy, axis=[1, 2, 3, 4])
