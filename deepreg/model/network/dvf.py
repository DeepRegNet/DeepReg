import tensorflow as tf

import deepreg.model.layer as layer
from deepreg.model.network.ddf import ddf_add_loss_metric
from deepreg.model.network.util import build_backbone, build_inputs


def dvf_forward(
    backbone: tf.keras.Model,
    moving_image: tf.Tensor,
    fixed_image: tf.Tensor,
    moving_label: (tf.Tensor, None),
    moving_image_size: tuple,
    fixed_image_size: tuple,
) -> [tf.Tensor, tf.Tensor, (tf.Tensor, None), tf.Tensor]:
    """
    Perform the network forward pass
    :param backbone: model architecture object, e.g. model.backbone.local_net
    :param moving_image: tensor of shape (batch, m_dim1, m_dim2, m_dim3)
    :param fixed_image:  tensor of shape (batch, f_dim1, f_dim2, f_dim3)
    :param moving_label: tensor of shape (batch, m_dim1, m_dim2, m_dim3) or None
    :param moving_image_size:
    :param fixed_image_size:
    :return: tuple(ddf, pred_fixed_image, pred_fixed_label, fixed_grid), where
    - dvf is the dense velocity field of shape (batch, m_dim1, m_dim2, m_dim3, 3)
    - pred_fixed_image is the predicted (warped) moving image of shape (batch, f_dim1, f_dim2, f_dim3)
    - pred_fixed_label is the predicted (warped) moving label of shape (batch, f_dim1, f_dim2, f_dim3)
    - fixed_grid is the grid of shape(f_dim1, f_dim2, f_dim3, 3)
    """

    # expand dims
    moving_image = tf.expand_dims(
        moving_image, axis=4
    )  # (batch, m_dim1, m_dim2, m_dim3, 1), need to be squeezed later for warping
    fixed_image = tf.expand_dims(
        fixed_image, axis=4
    )  # (batch, f_dim1, f_dim2, f_dim3, 1)

    # adjust moving image
    if moving_image_size != fixed_image_size:
        moving_image = layer.Resize3d(size=fixed_image_size)(
            inputs=moving_image
        )  # (batch, f_dim1, f_dim2, f_dim3, 1)

    # ddf
    inputs = tf.concat(
        [moving_image, fixed_image], axis=4
    )  # (batch, f_dim1, f_dim2, f_dim3, 2)
    dvf = backbone(inputs=inputs)  # (batch, f_dim1, f_dim2, f_dim3, 3)
    ddf = layer.IntDVF(fixed_image_size=fixed_image_size)(
        dvf
    )  # (batch, f_dim1, f_dim2, f_dim3, 3)

    # prediction, (batch, f_dim1, f_dim2, f_dim3)
    warping = layer.Warping(fixed_image_size=fixed_image_size)
    grid_fixed = tf.squeeze(warping.grid_ref, axis=0)  # (f_dim1, f_dim2, f_dim3, 3)
    pred_fixed_image = warping(inputs=[ddf, tf.squeeze(moving_image, axis=4)])
    pred_fixed_label = (
        warping(inputs=[ddf, moving_label]) if moving_label is not None else None
    )
    return dvf, ddf, pred_fixed_image, pred_fixed_label, grid_fixed


def build_dvf_model(
    moving_image_size: tuple,
    fixed_image_size: tuple,
    index_size: int,
    labeled: bool,
    batch_size: int,
    model_config: dict,
    loss_config: dict,
):
    """

    :param moving_image_size: (m_dim1, m_dim2, m_dim3)
    :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
    :param index_size:
    :param labeled:
    :param batch_size:
    :param model_config:
    :param loss_config:
    :return:
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
        method_name="dvf",
    )

    # forward
    dvf, ddf, pred_fixed_image, pred_fixed_label, grid_fixed = dvf_forward(
        backbone=backbone,
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_label=moving_label,
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
    )

    # build model
    if moving_label is None:  # unlabeled
        model = tf.keras.Model(
            inputs=dict(
                moving_image=moving_image, fixed_image=fixed_image, indices=indices
            ),
            outputs=dict(dvf=dvf, ddf=ddf),
            name="DVFRegModelWithoutLabel",
        )
    else:  # labeled
        model = tf.keras.Model(
            inputs=dict(
                moving_image=moving_image,
                fixed_image=fixed_image,
                indices=indices,
                moving_label=moving_label,
                fixed_label=fixed_label,
            ),
            outputs=dict(dvf=dvf, ddf=ddf, pred_fixed_label=pred_fixed_label),
            name="DVFRegModelWithLabel",
        )

    # loss and metric
    ddf_add_loss_metric(
        model=model,
        ddf=ddf,
        grid_fixed=grid_fixed,
        fixed_image=fixed_image,
        fixed_label=fixed_label,
        pred_fixed_image=pred_fixed_image,
        pred_fixed_label=pred_fixed_label,
        loss_config=loss_config,
    )

    return model
