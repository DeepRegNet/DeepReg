# coding=utf-8

"""
Module to build backbone modules based on passed inputs.
"""

import tensorflow as tf

import deepreg.model.loss.label as label_loss
from deepreg.registry import Registry


def build_backbone(
    image_size: tuple,
    out_channels: int,
    config: dict,
    method_name: str,
    registry: Registry,
) -> tf.keras.Model:
    """
    Backbone model accepts a single input of shape (batch, dim1, dim2, dim3, ch_in)
    and returns a single output of shape (batch, dim1, dim2, dim3, ch_out).

    :param image_size: tuple, dims of image, (dim1, dim2, dim3)
    :param out_channels: int, number of out channels, ch_out
    :param method_name: str, one of ddf, dvf and conditional
    :param config: dict, backbone configuration
    :param registry: the registry object having all registered classes
    :return: tf.keras.Model
    """
    if not (
        (isinstance(image_size, tuple) or isinstance(image_size, list))
        and len(image_size) == 3
    ):
        raise ValueError(f"image_size must be tuple of length 3, got {image_size}")

    if method_name in ["ddf", "dvf"]:
        out_activation = None
        # TODO try random init with smaller number
        out_kernel_initializer = "zeros"  # to ensure small ddf and dvf
    elif method_name in ["conditional"]:
        out_activation = "sigmoid"  # output is probability
        out_kernel_initializer = "glorot_uniform"
    elif method_name in ["affine"]:
        out_activation = None
        out_kernel_initializer = "zeros"
    else:
        raise ValueError(
            f"method name has to be one of ddf/dvf/conditional/affine, "
            f"got {method_name}"
        )

    backbone = registry.build_backbone(
        config=config,
        default_args=dict(
            image_size=image_size,
            out_channels=out_channels,
            out_kernel_initializer=out_kernel_initializer,
            out_activation=out_activation,
        ),
    )
    return backbone


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

    TODO do we absolutely need the batch_size in Input?

    :param moving_image_size: tuple, dims of moving images, (m_dim1, m_dim2, m_dim3)
    :param fixed_image_size: tuple, dims of fixed images, (f_dim1, f_dim2, f_dim3)
    :param index_size: int, dataset size (number of images)
    :param batch_size: int, mini-batch size
    :param labeled: Boolean, true if we have label data
    :return: 5 (if labeled=True) or 3 (if labeled=False) tf.keras.Input objects
    """
    moving_image = tf.keras.Input(
        shape=moving_image_size, batch_size=batch_size, name="moving_image"
    )  # (batch, m_dim1, m_dim2, m_dim3)
    fixed_image = tf.keras.Input(
        shape=fixed_image_size, batch_size=batch_size, name="fixed_image"
    )  # (batch, f_dim1, f_dim2, f_dim3)
    moving_label = (
        tf.keras.Input(
            shape=moving_image_size, batch_size=batch_size, name="moving_label"
        )
        if labeled
        else None
    )  # (batch, m_dim1, m_dim2, m_dim3)
    fixed_label = (
        tf.keras.Input(
            shape=fixed_image_size, batch_size=batch_size, name="fixed_label"
        )
        if labeled
        else None
    )  # (batch, m_dim1, m_dim2, m_dim3)
    indices = tf.keras.Input(
        shape=(index_size,), batch_size=batch_size, name="indices"
    )  # (batch, 2)
    return moving_image, fixed_image, moving_label, fixed_label, indices


def add_ddf_loss(
    model: tf.keras.Model,
    ddf: tf.Tensor,
    loss_config: dict,
    registry: Registry,
) -> tf.keras.Model:
    """
    Add regularization loss of ddf into model.

    :param model: tf.keras.Model
    :param ddf: tensor of shape (batch, m_dim1, m_dim2, m_dim3, 3)
    :param loss_config: config for loss
    :param registry: the registry object having all registered classes
    """
    if loss_config["regularization"]["weight"] <= 0:
        # TODO will refactor the way building models
        return model  # pragma: no cover
    config = loss_config["regularization"].copy()
    weight = config.pop("weight", 1)
    loss_reg = registry.build_loss(config=config)(inputs=ddf)
    weighted_loss_reg = loss_reg * weight
    model.add_loss(weighted_loss_reg)
    model.add_metric(loss_reg, name="loss/regularization", aggregation="mean")
    model.add_metric(
        weighted_loss_reg, name="loss/weighted_regularization", aggregation="mean"
    )
    return model


def add_image_loss(
    model: tf.keras.Model,
    fixed_image: tf.Tensor,
    pred_fixed_image: tf.Tensor,
    loss_config: dict,
    registry: Registry,
) -> tf.keras.Model:
    """
    Add image dissimilarity loss of ddf into model.

    :param model: tf.keras.Model
    :param fixed_image: tensor of shape (batch, f_dim1, f_dim2, f_dim3)
    :param pred_fixed_image: tensor of shape (batch, f_dim1, f_dim2, f_dim3)
    :param loss_config: config for loss
    :param registry: the registry object having all registered classes
    """
    if loss_config["image"]["weight"] <= 0:
        # TODO will refactor the way building models
        return model  # pragma: no cover
    config = loss_config["image"].copy()
    weight = config.pop("weight", 1)

    loss_image = registry.build_loss(config=config)(
        y_true=fixed_image,
        y_pred=pred_fixed_image,
    )
    weighted_loss_image = loss_image * weight
    model.add_loss(weighted_loss_image)
    model.add_metric(loss_image, name="loss/image_dissimilarity", aggregation="mean")
    model.add_metric(
        weighted_loss_image,
        name="loss/weighted_image_dissimilarity",
        aggregation="mean",
    )
    return model


def add_label_loss(
    model: tf.keras.Model,
    grid_fixed: tf.Tensor,
    fixed_label: (tf.Tensor, None),
    pred_fixed_label: (tf.Tensor, None),
    loss_config: dict,
    registry: Registry,
) -> tf.keras.Model:
    """
    Add label dissimilarity loss of ddf into model.

    :param model: tf.keras.Model
    :param grid_fixed: tensor of shape (f_dim1, f_dim2, f_dim3, 3)
    :param fixed_label: tensor of shape (batch, f_dim1, f_dim2, f_dim3)
    :param pred_fixed_label: tensor of shape (batch, f_dim1, f_dim2, f_dim3)
    :param loss_config: config for loss
    :param registry: the registry object having all registered classes
    """
    if fixed_label is None:
        # TODO will refactor the way building models
        return model  # pragma: no cover
    if loss_config["label"]["weight"] <= 0:
        # TODO will refactor the way building models
        return model  # pragma: no cover
    config = loss_config["label"].copy()
    weight = config.pop("weight", 1)
    loss_label = registry.build_loss(config=config)(
        y_true=fixed_label,
        y_pred=pred_fixed_label,
    )
    weighted_loss_label = loss_label * weight
    model.add_loss(weighted_loss_label)
    model.add_metric(loss_label, name="loss/label", aggregation="mean")
    model.add_metric(
        weighted_loss_label,
        name="loss/weighted_label",
        aggregation="mean",
    )

    # metrics
    dice_binary = label_loss.DiceScore(binary=True)(
        y_true=fixed_label, y_pred=pred_fixed_label
    )
    dice_float = label_loss.DiceScore(binary=False)(
        y_true=fixed_label, y_pred=pred_fixed_label
    )
    tre = label_loss.compute_centroid_distance(
        y_true=fixed_label, y_pred=pred_fixed_label, grid=grid_fixed
    )
    foreground_label = label_loss.foreground_proportion(y=fixed_label)
    foreground_pred = label_loss.foreground_proportion(y=pred_fixed_label)
    model.add_metric(dice_binary, name="metric/dice_binary", aggregation="mean")
    model.add_metric(dice_float, name="metric/dice_float", aggregation="mean")
    model.add_metric(tre, name="metric/tre", aggregation="mean")
    model.add_metric(
        foreground_label, name="metric/foreground_label", aggregation="mean"
    )
    model.add_metric(foreground_pred, name="metric/foreground_pred", aggregation="mean")
    return model
