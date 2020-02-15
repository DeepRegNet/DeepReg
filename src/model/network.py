import tensorflow as tf

import src.model.layer as layer
import src.model.loss as loss
from src.model.backbone.local_net import LocalModel


def build_backbone(moving_image_size, fixed_image_size, tf_model_config):
    if tf_model_config["name"] == "local":
        return LocalModel(moving_image_size=moving_image_size, fixed_image_size=fixed_image_size,
                          **tf_model_config["local"])
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

    # backbone
    backbone = build_backbone(moving_image_size=moving_image_size, fixed_image_size=fixed_image_size,
                              tf_model_config=tf_model_config)

    # ddf
    ddf = backbone(inputs=[moving_image, fixed_image])

    # prediction
    pred_fixed_label = layer.Warping(fixed_image_size=fixed_image_size)([ddf, moving_label])

    # build model
    reg_model = tf.keras.Model(inputs=[moving_image, fixed_image, moving_label],
                               outputs=pred_fixed_label,
                               name="RegModel")

    # add regularization loss
    reg_loss = tf.reduce_mean(loss.local_displacement_energy(ddf, **tf_loss_config["regularization"]))
    reg_model.add_loss(reg_loss)
    return reg_model
