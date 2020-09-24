import tensorflow as tf

from deepreg.model import layer, layer_util
from deepreg.model.network.util import (
    add_ddf_loss,
    add_image_loss,
    add_label_loss,
    build_backbone,
    build_inputs,
)


def ddf_dvf_forward(
    backbone: tf.keras.Model,
    moving_image: tf.Tensor,
    fixed_image: tf.Tensor,
    moving_label: (tf.Tensor, None),
    moving_image_size: tuple,
    fixed_image_size: tuple,
    output_dvf: bool,
) -> [(tf.Tensor, None), tf.Tensor, tf.Tensor, (tf.Tensor, None), tf.Tensor]:
    """
    Perform the network forward pass.

    :param backbone: model architecture object, e.g. model.backbone.local_net
    :param moving_image: tensor of shape (batch, m_dim1, m_dim2, m_dim3)
    :param fixed_image:  tensor of shape (batch, f_dim1, f_dim2, f_dim3)
    :param moving_label: tensor of shape (batch, m_dim1, m_dim2, m_dim3) or None
    :param moving_image_size: tuple like (m_dim1, m_dim2, m_dim3)
    :param fixed_image_size: tuple like (f_dim1, f_dim2, f_dim3)
    :param output_dvf: bool, if true, model outputs dvf, if false, model outputs ddf
    :return: (dvf, ddf, pred_fixed_image, pred_fixed_label, fixed_grid), where

      - dvf is the dense velocity field of shape (batch, f_dim1, f_dim2, f_dim3, 3)
      - ddf is the dense displacement field of shape (batch, f_dim1, f_dim2, f_dim3, 3)
      - pred_fixed_image is the predicted (warped) moving image of shape (batch, f_dim1, f_dim2, f_dim3)
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

    # adjust moving image
    moving_image = layer_util.resize3d(
        image=moving_image, size=fixed_image_size
    )  # (batch, f_dim1, f_dim2, f_dim3, 1)

    # ddf, dvf
    inputs = tf.concat(
        [moving_image, fixed_image], axis=4
    )  # (batch, f_dim1, f_dim2, f_dim3, 2)
    backbone_out = backbone(inputs=inputs)  # (batch, f_dim1, f_dim2, f_dim3, 3)
    if output_dvf:
        dvf = backbone_out  # (batch, f_dim1, f_dim2, f_dim3, 3)
        ddf = layer.IntDVF(fixed_image_size=fixed_image_size)(
            dvf
        )  # (batch, f_dim1, f_dim2, f_dim3, 3)
    else:
        dvf = None
        ddf = backbone_out  # (batch, f_dim1, f_dim2, f_dim3, 3)

    # prediction, (batch, f_dim1, f_dim2, f_dim3)
    warping = layer.Warping(fixed_image_size=fixed_image_size)
    grid_fixed = tf.squeeze(warping.grid_ref, axis=0)  # (f_dim1, f_dim2, f_dim3, 3)
    pred_fixed_image = warping(inputs=[ddf, tf.squeeze(moving_image, axis=4)])
    pred_fixed_label = (
        warping(inputs=[ddf, moving_label]) if moving_label is not None else None
    )
    return dvf, ddf, pred_fixed_image, pred_fixed_label, grid_fixed


def build_ddf_dvf_model(
    moving_image_size: tuple,
    fixed_image_size: tuple,
    index_size: int,
    labeled: bool,
    batch_size: int,
    model_config: dict,
    loss_config: dict,
) -> tf.keras.Model:
    """
    Build a model which outputs DDF/DVF.

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
        out_channels=3,
        model_config=model_config,
        method_name=model_config["method"],
    )

    # forward
    dvf, ddf, pred_fixed_image, pred_fixed_label, grid_fixed = ddf_dvf_forward(
        backbone=backbone,
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_label=moving_label,
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        output_dvf=model_config["method"] == "dvf",
    )

    # build model
    inputs = {
        "moving_image": moving_image,
        "fixed_image": fixed_image,
        "indices": indices,
    }
    outputs = {"ddf": ddf}
    if dvf is not None:
        outputs["dvf"] = dvf
    model_name = model_config["method"].upper() + "RegistrationModel"
    if moving_label is None:  # unlabeled
        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name=model_name + "WithoutLabel"
        )
    else:  # labeled
        inputs["moving_label"] = moving_label
        inputs["fixed_label"] = fixed_label
        outputs["pred_fixed_label"] = pred_fixed_label
        model = tf.keras.Model(
            inputs=inputs, outputs=outputs, name=model_name + "WithLabel"
        )

    # add loss and metric
    model = add_ddf_loss(model=model, ddf=ddf, loss_config=loss_config)
    model = add_image_loss(
        model=model,
        fixed_image=fixed_image,
        pred_fixed_image=pred_fixed_image,
        loss_config=loss_config,
    )
    model = add_label_loss(
        model=model,
        grid_fixed=grid_fixed,
        fixed_label=fixed_label,
        pred_fixed_label=pred_fixed_label,
        loss_config=loss_config,
    )

    return model
