"""
Module provides regularization energy functions for ddf.
"""
from typing import Callable

import tensorflow as tf


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


def compute_gradient_norm(ddf: tf.Tensor, l1: bool = False) -> tf.Tensor:
    """
    Calculate the L1/L2 norm of the first-order differentiation of ddf using central finite difference.

    :param ddf: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
    :param l1: bool true if calculate L1 norm, otherwise L2 norm
    :return: shape = (batch, )
    """
    # first order gradient
    # (batch, m_dim1-2, m_dim2-2, m_dim3-2, 3)
    dfdx = gradient_dxyz(ddf, gradient_dx)
    dfdy = gradient_dxyz(ddf, gradient_dy)
    dfdz = gradient_dxyz(ddf, gradient_dz)
    if l1:
        norms = tf.abs(dfdx) + tf.abs(dfdy) + tf.abs(dfdz)
    else:
        norms = dfdx ** 2 + dfdy ** 2 + dfdz ** 2
    return tf.reduce_mean(norms, [1, 2, 3, 4])  # (batch,)


def compute_bending_energy(ddf: tf.Tensor) -> tf.Tensor:
    """
    Calculate the bending energy based on second-order differentiation of ddf using central finite difference.

    :param ddf: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
    :return: shape = (batch, )
    """
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
    return tf.reduce_mean(energy, [1, 2, 3, 4])


def local_displacement_energy(ddf: tf.Tensor, energy_type: str, **kwargs) -> tf.Tensor:
    """
    Calculate the displacement energy of the ddf based on finite difference.

    :param ddf: shape = (batch, m_dim1, m_dim2, m_dim3, 3)
    :param energy_type: type of the energy
    :param kwargs: absorb additional arguments
    :return: shape = (batch,)
    """

    if energy_type == "bending":
        return compute_bending_energy(ddf)
    elif energy_type == "gradient-l2":
        return compute_gradient_norm(ddf, l1=False)
    elif energy_type == "gradient-l1":
        return compute_gradient_norm(ddf, l1=True)
    else:
        raise ValueError(f"Unknown energy_type {energy_type}.")
