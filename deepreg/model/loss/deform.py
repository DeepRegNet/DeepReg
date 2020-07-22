"""
Module for calculating local displacement energy.
"""
import tensorflow as tf


def gradient_dx(fv):
    """
    Function to calculate gradients on x-axis of a 3D tensor.

    :param fv: a 3D tensor

    :return: the gradients on x-axis
    """
    return (fv[:, 2:, 1:-1, 1:-1] - fv[:, :-2, 1:-1, 1:-1]) / 2


def gradient_dy(fv):
    """
    Function to calculate gradients on y-axis of a 3D tensor.

    :param fv: a 3D tensor

    :return: the gradients on y-axis
    """
    return (fv[:, 1:-1, 2:, 1:-1] - fv[:, 1:-1, :-2, 1:-1]) / 2


def gradient_dz(fv):
    """
    Function to calculate gradients on z-axis of a 3D tensor.

    :param fv: a 3D tensor

    :return: the gradients on z-axis
    """
    return (fv[:, 1:-1, 1:-1, 2:] - fv[:, 1:-1, 1:-1, :-2]) / 2


def gradient_txyz(Txyz, fn):
    """
    Function to calculate gradients on a 4D tensor (DDF).

    :param Txyz: a DDF
    :param fn: a gradient calculating function

    :return: the gradients of the ddf on each channel
    """
    return tf.stack([fn(Txyz[..., i]) for i in [0, 1, 2]], axis=4)


def compute_gradient_norm(displacement, l1=False):
    """
    Function to calculate l1/l2-norm for a DDF.

    :param displacement: a DDF
    :param l1: choose to calculate l1 or l2-norm

    :return: the l1 or l2-norm for a DDF
    """
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    if l1:
        norms = tf.abs(dTdx) + tf.abs(dTdy) + tf.abs(dTdz)
    else:
        norms = dTdx ** 2 + dTdy ** 2 + dTdz ** 2
    return tf.reduce_mean(norms, [1, 2, 3, 4])


def compute_bending_energy(displacement):
    """
    Function to calculate the bending energy of a DDF.

    :param displacement: a DDF

    :return: the bending energy for a DDF
    """
    dTdx = gradient_txyz(displacement, gradient_dx)
    dTdy = gradient_txyz(displacement, gradient_dy)
    dTdz = gradient_txyz(displacement, gradient_dz)
    dTdxx = gradient_txyz(dTdx, gradient_dx)
    dTdyy = gradient_txyz(dTdy, gradient_dy)
    dTdzz = gradient_txyz(dTdz, gradient_dz)
    dTdxy = gradient_txyz(dTdx, gradient_dy)
    dTdyz = gradient_txyz(dTdy, gradient_dz)
    dTdxz = gradient_txyz(dTdx, gradient_dz)
    return tf.reduce_mean(
        dTdxx ** 2
        + dTdyy ** 2
        + dTdzz ** 2
        + 2 * dTdxy ** 2
        + 2 * dTdxz ** 2
        + 2 * dTdyz ** 2,
        [1, 2, 3, 4],
    )


def local_displacement_energy(ddf, energy_type, **kwargs):
    """
    Function to calculate the energy of a DDF.

    :param ddf: a DDF
    :param energy_type: bending/gradient-l2/gradient-l1

    :return: the energy for a DDF
    """
    if energy_type == "bending":
        return compute_bending_energy(ddf)
    elif energy_type == "gradient-l2":
        return compute_gradient_norm(ddf)
    elif energy_type == "gradient-l1":
        return compute_gradient_norm(ddf, l1=True)
    else:
        raise ValueError("Unknown regularizer.")
