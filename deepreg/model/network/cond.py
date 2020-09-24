import tensorflow as tf

from deepreg.model import layer, layer_util
from deepreg.model.network.util import add_label_loss, build_backbone, build_inputs


def conditional_forward(
    backbone: tf.keras.Model,
    moving_image: tf.Tensor,
    fixed_image: tf.Tensor,
    moving_label: (tf.Tensor, None),
    moving_image_size: tuple,
    fixed_image_size: tuple,
) -> [tf.Tensor, tf.Tensor]:
    """
    Perform the network forward pass.

    :param backbone: model architecture object, e.g. model.backbone.local_net
    :param moving_image: tensor of shape (batch, m_dim1, m_dim2, m_dim3)
    :param fixed_image:  tensor of shape (batch, f_dim1, f_dim2, f_dim3)
    :param moving_label: tensor of shape (batch, m_dim1, m_dim2, m_dim3) or None
    :param moving_image_size: tuple like (m_dim1, m_dim2, m_dim3)
    :param fixed_image_size: tuple like (f_dim1, f_dim2, f_dim3)
    :return: (pred_fixed_label, fixed_grid), where

      - pred_fixed_label is the predicted (warped) moving label of shape (batch, f_dim1, f_dim2, f_dim3)
      - fixed_grid is the grid of shape(f_dim1, f_dim2, f_dim3, 3)
    """

    # expand dims
    # need to be squeezed later for warping
    moving_image = tf.expand_dims(
        moving_image, axis=4
    )  # (batch, m_dim1, m_dim2, m_dim3, 1)
    fixed_image = tf.expand_dims(
        fixed_image, axis=4
    )  # (batch, f_dim1, f_dim2, f_dim3, 1)
    moving_label = tf.expand_dims(
        moving_label, axis=4
    )  # (batch, m_dim1, m_dim2, m_dim3, 1)

    # adjust moving image
    if moving_image_size != fixed_image_size:
        moving_image = layer_util.resize3d(
            image=moving_image, size=fixed_image_size
        )  # (batch, f_dim1, f_dim2, f_dim3, 1)
        moving_label = layer_util.resize3d(
            image=moving_label, size=fixed_image_size
        )  # (batch, f_dim1, f_dim2, f_dim3, 1)

    # conditional
    inputs = tf.concat(
        [moving_image, fixed_image, moving_label], axis=4
    )  # (batch, f_dim1, f_dim2, f_dim3, 3)
    pred_fixed_label = backbone(inputs=inputs)  # (batch, f_dim1, f_dim2, f_dim3, 1)
    pred_fixed_label = tf.squeeze(
        pred_fixed_label, axis=4
    )  # (batch, f_dim1, f_dim2, f_dim3)

    warping = layer.Warping(fixed_image_size=fixed_image_size)
    grid_fixed = tf.squeeze(warping.grid_ref, axis=0)  # (f_dim1, f_dim2, f_dim3, 3)

    return pred_fixed_label, grid_fixed


def build_conditional_model(
    moving_image_size: tuple,
    fixed_image_size: tuple,
    index_size: int,
    labeled: bool,
    batch_size: int,
    model_config: dict,
    loss_config: dict,
) -> tf.keras.Model:
    """
    Build a model which outputs predicted fixed label.

    :param moving_image_size: (m_dim1, m_dim2, m_dim3)
    :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
    :param index_size: int, the number of indices for identifying a sample
    :param labeled: bool, indicating if the data is labeled
    :param batch_size: int, size of mini-batch
    :param model_config: config for the model
    :param loss_config: config for the loss
    :return: the built tf.keras.Model
    """
    # inputs
    (moving_image, fixed_image, moving_label, fixed_label, indices) = build_inputs(
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        index_size=index_size,
        batch_size=batch_size,
        labeled=labeled,
    )

    # backbone
    backbone = build_backbone(
        image_size=fixed_image_size,
        out_channels=1,
        model_config=model_config,
        method_name=model_config["method"],
    )

    # prediction
    pred_fixed_label, grid_fixed = conditional_forward(
        backbone=backbone,
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_label=moving_label,
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
    )  # (batch, f_dim1, f_dim2, f_dim3)

    # build model
    inputs = {
        "moving_image": moving_image,
        "fixed_image": fixed_image,
        "moving_label": moving_label,
        "fixed_label": fixed_label,
        "indices": indices,
    }
    outputs = {"pred_fixed_label": pred_fixed_label}
    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="ConditionalRegistrationModel"
    )

    # loss and metric
    model = add_label_loss(
        model=model,
        grid_fixed=grid_fixed,
        fixed_label=fixed_label,
        pred_fixed_label=pred_fixed_label,
        loss_config=loss_config,
    )

    return model
