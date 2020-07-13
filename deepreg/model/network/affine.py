import tensorflow as tf

import deepreg.model.layer as layer
import deepreg.model.loss.deform as deform_loss
import deepreg.model.loss.image as image_loss
import deepreg.model.loss.label as label_loss
from deepreg.model.network.util import build_backbone, build_inputs


def build_affine_model(
    moving_image_size,
    fixed_image_size,
    index_size,
    labeled,
    batch_size,
    model_config,
    loss_config,
):
    """
    Build the model if the output is an affine-based DDF (dense displacement field)
    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param batch_size: minibatch size
    :param model_config: model configuration, e.g. dictionary return from parser.yaml.load
    :param loss_config: loss configuration, e.g. dictionary return from parser.yaml.load
    :return: the built tf.keras.Model
    """

    def forward(_backbone, _moving_image, _moving_label, _fixed_image):
        """
        Perform the network forward pass
        :param _backbone: model architecture object, e.g. return from model.backbone.global_net
        :param _moving_image: [batch, m_dim1, m_dim2, m_dim3]
        :param _moving_label: [batch, m_dim1, m_dim2, m_dim3]
        :param _fixed_image:  [batch, f_dim1, f_dim2, f_dim3]
        :return: tuple(_affine, _ddf, _pred_fixed_image, _pred_fixed_label)
            WHERE
            str _affine is the affine transformation matrix predicted by the network [batch, 3, 4]
            str _ddf is the dense displacement field [batch, m_dim1, m_dim2, m_dim3, 3]
            str _pred_fixed_image is the predicted (warped) moving image [batch, f_dim1, f_dim2, f_dim3]
            str _pred_fixed_label is the predicted (warped) moving label [batch, f_dim1, f_dim2, f_dim3]
        """
        # ddf
        backbone_input = tf.concat(
            [
                layer.Resize3d(size=fixed_image_size)(
                    inputs=tf.expand_dims(_moving_image, axis=4)
                ),
                tf.expand_dims(_fixed_image, axis=4),
            ],
            axis=4,
        )  # [batch, f_dim1, f_dim2, f_dim3, 2]

        _ddf = _backbone(inputs=backbone_input)  # [batch, f_dim1, f_dim2, f_dim3, 3]
        _affine = _backbone.theta

        # prediction image ang label shape = [batch, f_dim1, f_dim2, f_dim3]
        _pred_fixed_image = layer.Warping(fixed_image_size=fixed_image_size)(
            [_ddf, _moving_image]
        )
        _pred_fixed_label = layer.Warping(fixed_image_size=fixed_image_size)(
            [_ddf, _moving_label]
        )
        return _affine, _ddf, _pred_fixed_image, _pred_fixed_label

    def add_loss_metric(
        _fixed_image, _pred_fixed_image, _ddf, _fixed_label, _pred_fixed_label, suffix
    ):
        """
        Configure and add the training loss, including image, label and deformation regularisation
        :param _fixed_image:      [batch, f_dim1, f_dim2, f_dim3]
        :param _pred_fixed_image: [batch, f_dim1, f_dim2, f_dim3]
        :param _ddf:              [batch, f_dim1, f_dim2, f_dim3, 3]
        :param _fixed_label:      [batch, f_dim1, f_dim2, f_dim3]
        :param _pred_fixed_label: [batch, f_dim1, f_dim2, f_dim3]
        :param suffix: string reserved or extra information
        :return: tf.keras.Model with loss and metric added
        """
        # image loss
        if loss_config["similarity"]["image"]["weight"] > 0:
            loss_image = tf.reduce_mean(
                image_loss.similarity_fn(
                    y_true=_fixed_image,
                    y_pred=_pred_fixed_image,
                    **loss_config["similarity"]["image"],
                )
            )
            weighted_loss_image = (
                loss_image * loss_config["similarity"]["image"]["weight"]
            )
            model.add_loss(weighted_loss_image)
            model.add_metric(
                loss_image, name="loss/image_similarity" + suffix, aggregation="mean"
            )
            model.add_metric(
                weighted_loss_image,
                name="loss/weighted_image_similarity" + suffix,
                aggregation="mean",
            )

        # label loss
        if _fixed_label is not None:
            label_loss_fn = label_loss.get_similarity_fn(
                config=loss_config["similarity"]["label"]
            )
            loss_label = label_loss_fn(y_true=_fixed_label, y_pred=_pred_fixed_label)
            model.add_loss(loss_label)
            model.add_metric(loss_label, name="loss/label" + suffix, aggregation="mean")

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
        image_size=fixed_image_size, out_channels=3, model_config=model_config, model_name=model_config["method"]
    )

    # forward
    affine, ddf, pred_fixed_image, pred_fixed_label = forward(
        _backbone=backbone,
        _moving_image=moving_image,
        _moving_label=moving_label,
        _fixed_image=fixed_image,
    )

    # build model
    model = tf.keras.Model(
        inputs=[moving_image, fixed_image, moving_label, indices],
        outputs=[pred_fixed_label],
        name="DDFRegModel",
    )

    model.ddf = ddf
    model.affine = affine

    # loss and metric
    add_loss_metric(
        _fixed_image=fixed_image,
        _pred_fixed_image=pred_fixed_image,
        _ddf=ddf,
        _fixed_label=None,
        _pred_fixed_label=None,
        suffix="",
    )
    return model
