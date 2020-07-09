# coding=utf-8

"""
Module to build backbone modules based on passed inputs.
"""

import tensorflow as tf

from deepreg.model.backbone.global_net import GlobalNet
from deepreg.model.backbone.local_net import LocalNet
from deepreg.model.backbone.u_net import UNet


def build_backbone(
    image_size: tuple, out_channels: int, model_config: dict, method_name: str
) -> tf.keras.Model:
    """
    Backbone model accepts a single input of shape (batch, dim1, dim2, dim3, ch_in)
    and returns a single output of shape (batch, dim1, dim2, dim3, ch_out)

    :param image_size: tuple, dims of image, (dim1, dim2, dim3)
    :param out_channels: int, number of out channels, ch_out
    :param method_name: str, one of ddf | dvf | conditional
    :param model_config: dict, model configuration, returned from parser.yaml.load
    :return: tf.keras.Model with shape (batch, dim1, dim2, dim3, ch_out)
    """
    if method_name not in ["ddf", "dvf", "conditional"]:
        raise ValueError(
            "method name has to be one of ddf / dvf / conditional in build_backbone, "
            "got {}".format(method_name)
        )

    if method_name in ["ddf", "dvf"]:
        out_activation = None
        # TODO try random init with smaller number
        out_kernel_initializer = "zeros"  # to ensure small ddf and dvf
    elif method_name in ["conditional"]:
        out_activation = "sigmoid"  # output is probability
        out_kernel_initializer = "glorot_uniform"
    else:
        raise ValueError("Unknown method name {}".format(method_name))

    if model_config["backbone"] == "local":
        return LocalNet(
            image_size=image_size,
            out_channels=out_channels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            **model_config["local"],
        )
    elif model_config["backbone"] == "global":
        return GlobalNet(
            image_size=image_size,
            out_channels=out_channels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            **model_config["global"],
        )
    elif model_config["backbone"] == "unet":
        return UNet(
            image_size=image_size,
            out_channels=out_channels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
            **model_config["unet"],
        )
    else:
        raise ValueError("Unknown model name")


def build_inputs(
    moving_image_size: tuple,
    fixed_image_size: tuple,
    index_size: int,
    batch_size: int,
    labeled: bool,
) -> [tf.keras.Input, tf.keras.Input, tf.keras.Input, tf.keras.Input, tf.keras.Input]:
    """
    Configure a pair of moving and fixed images and a pair of
    moving and fixed labels as model input
    and returns model input tf.keras.Input

    :param moving_image_size: tuple, dims of moving images, [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: tuple, dims of fixed images, [f_dim1, f_dim2, f_dim3]
    :param index_size: int, dataset size (number of images)
    :param batch_size: int, mini-batch size
    :param labeled: Boolean, true if we have label data
    :return: 5 (if labeled=True) or 3 (if labeled=False) tf.keras.Input objects
    """
    moving_image = tf.keras.Input(
        shape=(*moving_image_size,), batch_size=batch_size, name="moving_image"
    )  # (batch, m_dim1, m_dim2, m_dim3)
    fixed_image = tf.keras.Input(
        shape=(*fixed_image_size,), batch_size=batch_size, name="fixed_image"
    )  # (batch, f_dim1, f_dim2, f_dim3)
    moving_label = (
        tf.keras.Input(
            shape=(*moving_image_size,), batch_size=batch_size, name="moving_label"
        )
        if labeled
        else None
    )  # (batch, m_dim1, m_dim2, m_dim3)
    fixed_label = (
        tf.keras.Input(
            shape=(*fixed_image_size,), batch_size=batch_size, name="fixed_label"
        )
        if labeled
        else None
    )  # (batch, m_dim1, m_dim2, m_dim3)
    indices = tf.keras.Input(
        shape=(index_size,), batch_size=batch_size, name="indices"
    )  # (batch, 2)
    return moving_image, fixed_image, moving_label, fixed_label, indices
