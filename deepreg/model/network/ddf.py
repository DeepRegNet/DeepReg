import tensorflow as tf

import deepreg.model.layer as layer
import deepreg.model.loss.deform as deform_loss
import deepreg.model.loss.image as image_loss
import deepreg.model.loss.label as label_loss
from deepreg.model.network.util import build_backbone, build_inputs


def ddf_forward(
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
    - ddf is the dense displacement field of shape (batch, m_dim1, m_dim2, m_dim3, 3)
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
    ddf = backbone(inputs=inputs)  # (batch, f_dim1, f_dim2, f_dim3, 3)

    # prediction, (batch, f_dim1, f_dim2, f_dim3)
    warping = layer.Warping(fixed_image_size=fixed_image_size)
    grid_fixed = tf.squeeze(warping.grid_ref, axis=0)  # (f_dim1, f_dim2, f_dim3, 3)
    pred_fixed_image = warping(inputs=[ddf, tf.squeeze(moving_image, axis=4)])
    pred_fixed_label = (
        warping(inputs=[ddf, moving_label]) if moving_label is not None else None
    )
    return ddf, pred_fixed_image, pred_fixed_label, grid_fixed


def ddf_add_loss_metric(
    model: tf.keras.Model,
    ddf: tf.Tensor,
    grid_fixed: tf.Tensor,
    fixed_image: tf.Tensor,
    fixed_label: (tf.Tensor, None),
    pred_fixed_image: tf.Tensor,
    pred_fixed_label: (tf.Tensor, None),
    loss_config: dict,
):
    """
    Configure and add the training loss, including image and deformation regularisation,
    label loss is added using when compiling the model.
    :param model:
    :param ddf:              (batch, m_dim1, m_dim2, m_dim3, 3)
    :param grid_fixed:       (f_dim1, f_dim2, f_dim3, 3)
    :param fixed_image:      (batch, f_dim1, f_dim2, f_dim3)
    :param fixed_label:      (batch, f_dim1, f_dim2, f_dim3)
    :param pred_fixed_image: (batch, f_dim1, f_dim2, f_dim3)
    :param pred_fixed_label: (batch, f_dim1, f_dim2, f_dim3)
    :param loss_config:
    :return:
    """
    # regularization loss on ddf
    loss_reg = tf.reduce_mean(
        deform_loss.local_displacement_energy(ddf, **loss_config["regularization"])
    )
    weighted_loss_reg = loss_reg * loss_config["regularization"]["weight"]
    model.add_loss(weighted_loss_reg)
    model.add_metric(loss_reg, name="loss/regularization", aggregation="mean")
    model.add_metric(
        weighted_loss_reg, name="loss/weighted_regularization", aggregation="mean"
    )

    # image loss
    if loss_config["dissimilarity"]["image"]["weight"] > 0:
        loss_image = tf.reduce_mean(
            image_loss.dissimilarity_fn(
                y_true=fixed_image,
                y_pred=pred_fixed_image,
                **loss_config["dissimilarity"]["image"],
            )
        )
        weighted_loss_image = (
            loss_image * loss_config["dissimilarity"]["image"]["weight"]
        )
        model.add_loss(weighted_loss_image)
        model.add_metric(
            loss_image, name="loss/image_dissimilarity", aggregation="mean"
        )
        model.add_metric(
            weighted_loss_image,
            name="loss/weighted_image_dissimilarity",
            aggregation="mean",
        )

    # label loss
    if fixed_label is not None:
        loss_label = tf.reduce_mean(
            label_loss.get_dissimilarity_fn(
                config=loss_config["dissimilarity"]["label"]
            )(y_true=fixed_label, y_pred=pred_fixed_label)
        )
        weighted_loss_label = loss_label
        model.add_loss(weighted_loss_label)
        model.add_metric(
            loss_label, name="loss/label_dissimilarity", aggregation="mean"
        )
        model.add_metric(
            weighted_loss_label,
            name="loss/weighted_label_dissimilarity",
            aggregation="mean",
        )

        # metrics
        dice_binary = label_loss.dice_score(
            y_true=fixed_label, y_pred=pred_fixed_label, binary=True
        )
        dice_float = label_loss.dice_score(
            y_true=fixed_label, y_pred=pred_fixed_label, binary=False
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
        model.add_metric(
            foreground_pred, name="metric/foreground_pred", aggregation="mean"
        )


def build_ddf_model(
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
        method_name="ddf",
    )

    # forward
    ddf, pred_fixed_image, pred_fixed_label, grid_fixed = ddf_forward(
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
            outputs=dict(ddf=ddf),
            name="DDFRegModelWithoutLabel",
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
            outputs=dict(ddf=ddf, pred_fixed_label=pred_fixed_label),
            name="DDFRegModelWithLabel",
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
