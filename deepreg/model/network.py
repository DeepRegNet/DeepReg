import tensorflow as tf

import deepreg.model.layer as layer
import deepreg.model.loss.deform
import deepreg.model.loss.image as image_loss
from deepreg.model.backbone.local_net import LocalNet
from deepreg.model.backbone.u_net import UNet


def build_backbone(image_size, out_channels, tf_model_config):
    """
    backbone model accepts a single input of shape [batch, dim1, dim2, dim3, ch_in]
               and returns a single output of shape [batch, dim1, dim2, dim3, ch_out]
    :param image_size: [dim1, dim2, dim3]
    :param out_channels: ch_out
    :param tf_model_config:
    :return:
    """

    # no activation
    if tf_model_config["backbone"]["out_activation"] == "":
        tf_model_config["backbone"]["out_activation"] = None

    if tf_model_config["backbone"]["name"] == "local":
        return LocalNet(image_size=image_size, out_channels=out_channels,
                        out_kernel_initializer=tf_model_config["backbone"]["out_kernel_initializer"],
                        out_activation=tf_model_config["backbone"]["out_activation"],
                        **tf_model_config["local"])
    elif tf_model_config["backbone"]["name"] == "unet":
        return UNet(image_size=image_size, out_channels=out_channels,
                    out_kernel_initializer=tf_model_config["backbone"]["out_kernel_initializer"],
                    out_activation=tf_model_config["backbone"]["out_activation"],
                    **tf_model_config["unet"])
    else:
        raise ValueError("Unknown model name")


def build_inputs(moving_image_size, fixed_image_size, index_size, batch_size):
    moving_image = tf.keras.Input(shape=(*moving_image_size,), batch_size=batch_size,
                                  name="moving_image")  # [batch, m_dim1, m_dim2, m_dim3]
    fixed_image = tf.keras.Input(shape=(*fixed_image_size,), batch_size=batch_size,
                                 name="fixed_image")  # [batch, f_dim1, f_dim2, f_dim3]
    moving_label = tf.keras.Input(shape=(*moving_image_size,), batch_size=batch_size,
                                  name="moving_label")  # [batch, m_dim1, m_dim2, m_dim3]
    indices = tf.keras.Input(shape=(index_size,), batch_size=batch_size,
                             name="indices")  # [batch, 2]
    return moving_image, fixed_image, moving_label, indices


def build_ddf_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config, tf_loss_config):
    """

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param batch_size:
    :param tf_model_config:
    :param tf_loss_config:
    :return:
    """
    # inputs
    moving_image, fixed_image, moving_label, indices = build_inputs(
        moving_image_size, fixed_image_size, index_size, batch_size)

    backbone_input = tf.concat([layer.Resize3d(size=fixed_image_size)(inputs=tf.expand_dims(moving_image, axis=4)),
                                tf.expand_dims(fixed_image, axis=4)],
                               axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 2]

    # backbone
    backbone = build_backbone(image_size=fixed_image_size, out_channels=3,
                              tf_model_config=tf_model_config)

    # ddf
    ddf = backbone(inputs=backbone_input)  # [batch, f_dim1, f_dim2, f_dim3, 3]

    # prediction image ang label shape = [batch, f_dim1, f_dim2, f_dim3]
    pred_fixed_image = layer.Warping(fixed_image_size=fixed_image_size)([ddf, moving_image])
    pred_fixed_label = layer.Warping(fixed_image_size=fixed_image_size)([ddf, moving_label])

    # build model
    model = tf.keras.Model(inputs=[moving_image, fixed_image, moving_label, indices],
                           outputs=[pred_fixed_label],
                           name="DDFRegModel")
    model.ddf = ddf

    # image loss
    if tf_loss_config["similarity"]["image"]["weight"] > 0:
        loss_image = tf.reduce_mean(image_loss.similarity_fn(
            y_true=fixed_image, y_pred=pred_fixed_image,
            **tf_loss_config["similarity"]["image"]))
        weighted_loss_image = loss_image * tf_loss_config["similarity"]["image"]["weight"]
        model.add_loss(weighted_loss_image)
        model.add_metric(loss_image, name="loss/image_similarity", aggregation="mean")
        model.add_metric(weighted_loss_image, name="loss/weighted_image_similarity", aggregation="mean")

    # regularization loss
    loss_reg = tf.reduce_mean(
        deepreg.model.loss.deform.local_displacement_energy(ddf, **tf_loss_config["regularization"]))
    weighted_loss_reg = loss_reg * tf_loss_config["regularization"]["weight"]
    model.add_loss(weighted_loss_reg)
    model.add_metric(loss_reg, name="loss/regularization", aggregation="mean")
    model.add_metric(weighted_loss_reg, name="loss/weighted_regularization", aggregation="mean")

    return model


def build_cond_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config, tf_loss_config):
    """

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param batch_size:
    :param tf_model_config:
    :param tf_loss_config:
    :return:
    """
    # inputs
    moving_image, fixed_image, moving_label, indices = build_inputs(
        moving_image_size, fixed_image_size, index_size, batch_size)
    backbone_input = tf.concat([layer.Resize3d(size=fixed_image_size)(inputs=tf.expand_dims(moving_image, axis=4)),
                                tf.expand_dims(fixed_image, axis=4),
                                layer.Resize3d(size=fixed_image_size)(inputs=tf.expand_dims(moving_label, axis=4)),
                                ],
                               axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 3]

    # backbone
    backbone = build_backbone(image_size=fixed_image_size, out_channels=1,
                              tf_model_config=tf_model_config)

    # prediction
    pred_fixed_label = backbone(inputs=backbone_input)  # [batch, f_dim1, f_dim2, f_dim3, 1]
    pred_fixed_label = tf.squeeze(pred_fixed_label, axis=4)

    # build model
    model = tf.keras.Model(inputs=[moving_image, fixed_image, moving_label, indices],
                           outputs=[pred_fixed_label],
                           name="CondRegModel")

    return model


def build_seg_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config, tf_loss_config):
    """

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param batch_size:
    :param tf_model_config:
    :param tf_loss_config:
    :return:
    """
    # inputs
    moving_image, fixed_image, moving_label, indices = build_inputs(
        moving_image_size, fixed_image_size, index_size, batch_size)
    backbone_input = tf.expand_dims(fixed_image, axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 3]

    # backbone
    backbone = build_backbone(image_size=fixed_image_size, out_channels=1,
                              tf_model_config=tf_model_config)

    # prediction
    pred_fixed_label = backbone(inputs=backbone_input)  # [batch, f_dim1, f_dim2, f_dim3, 1]
    pred_fixed_label = tf.squeeze(pred_fixed_label, axis=4)

    # build model
    model = tf.keras.Model(inputs=[moving_image, fixed_image, moving_label, indices],
                           outputs=[pred_fixed_label],
                           name="SegModel")

    return model


def build_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config, tf_loss_config):
    if tf_model_config["method"] == "ddf":
        return build_ddf_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config,
                               tf_loss_config)
    elif tf_model_config["method"] == "conditional":
        return build_cond_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config,
                                tf_loss_config)
    elif tf_model_config["method"] == "seg":
        return build_seg_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config,
                               tf_loss_config)
    else:
        raise ValueError("Unknown model method")
