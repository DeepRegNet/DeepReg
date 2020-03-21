import tensorflow as tf

import src.model.layer as layer
import src.model.loss.deform
import src.model.loss.image as image_loss
from src.model.backbone.local_net import LocalNet
from src.model.backbone.u_net import UNet


def build_backbone(moving_image_size, fixed_image_size, tf_model_config):
    if tf_model_config["name"] == "local":
        return LocalNet(moving_image_size=moving_image_size, fixed_image_size=fixed_image_size,
                        **tf_model_config["local"])
    elif tf_model_config["name"] == "unet":
        return UNet(moving_image_size=moving_image_size, fixed_image_size=fixed_image_size,
                    **tf_model_config["unet"])
    else:
        raise ValueError("Unknown model name")


def build_model(moving_image_size, fixed_image_size, batch_size, tf_model_config, tf_loss_config):
    """

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param batch_size:
    :param tf_model_config:
    :param tf_loss_config:
    :return:
    """
    # inputs
    moving_image = tf.keras.Input(shape=(*moving_image_size,), batch_size=batch_size,
                                  name="moving_image")  # [batch, m_dim1, m_dim2, m_dim3]
    fixed_image = tf.keras.Input(shape=(*fixed_image_size,), batch_size=batch_size,
                                 name="fixed_image")  # [batch, f_dim1, f_dim2, f_dim3]
    moving_label = tf.keras.Input(shape=(*moving_image_size,), batch_size=batch_size,
                                  name="moving_label")  # [batch, m_dim1, m_dim2, m_dim3]
    indices = tf.keras.Input(shape=(2,), batch_size=batch_size,
                             name="indices")  # [batch, 2]
    # backbone
    backbone = build_backbone(moving_image_size=moving_image_size, fixed_image_size=fixed_image_size,
                              tf_model_config=tf_model_config)

    # ddf
    ddf = backbone(inputs=[moving_image, fixed_image])

    # prediction
    pred_fixed_image = layer.Warping(fixed_image_size=fixed_image_size)([ddf, moving_image])
    pred_fixed_label = layer.Warping(fixed_image_size=fixed_image_size)([ddf, moving_label])

    # build model
    reg_model = tf.keras.Model(inputs=[moving_image, fixed_image, moving_label, indices],
                               outputs=[pred_fixed_label],
                               name="RegModel")

    # image loss
    if tf_loss_config["similarity"]["image"]["weight"] > 0:
        loss_image = tf.reduce_mean(image_loss.similarity_fn(
            y_true=fixed_image, y_pred=pred_fixed_image,
            **tf_loss_config["similarity"]["image"]))
        weighted_loss_image = loss_image * tf_loss_config["similarity"]["image"]["weight"]
        reg_model.add_loss(weighted_loss_image)
        reg_model.add_metric(loss_image, name="loss/image_similarity", aggregation="mean")
        reg_model.add_metric(weighted_loss_image, name="loss/weighted_image_similarity", aggregation="mean")

    # regularization loss
    loss_reg = tf.reduce_mean(
        src.model.loss.deform.local_displacement_energy(ddf, **tf_loss_config["regularization"]))
    weighted_loss_reg = loss_reg * tf_loss_config["regularization"]["weight"]
    reg_model.add_loss(weighted_loss_reg)
    reg_model.add_metric(loss_reg, name="loss/regularization", aggregation="mean")
    reg_model.add_metric(weighted_loss_reg, name="loss/weighted_regularization", aggregation="mean")

    return reg_model
