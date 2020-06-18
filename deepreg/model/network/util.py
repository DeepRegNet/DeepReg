import tensorflow as tf

from deepreg.model.backbone.local_net import LocalNet
from deepreg.model.backbone.u_net import UNet
from deepreg.model.backbone.global_net import GlobalNet


def build_backbone(image_size: tuple, out_channels: int, tf_model_config: dict) -> tf.keras.Model:
    """
    backbone model accepts a single input of shape (batch, dim1, dim2, dim3, ch_in)
    and returns a single output of shape (batch, dim1, dim2, dim3, ch_out)
    :param image_size: (dim1, dim2, dim3)
    :param out_channels: ch_out
    :param tf_model_config:model configuration, e.g. dictionary return from parser.yaml.load
    :return:
    """

    if tf_model_config["backbone"]["out_activation"] == "":  # no activation
        tf_model_config["backbone"]["out_activation"] = None

    if tf_model_config["backbone"]["name"] == "local":
        return LocalNet(image_size=image_size, out_channels=out_channels,
                        out_kernel_initializer=tf_model_config["backbone"]["out_kernel_initializer"],
                        out_activation=tf_model_config["backbone"]["out_activation"],
                        **tf_model_config["local"])
    elif tf_model_config["backbone"]["name"] == "global":
        return GlobalNet(image_size=image_size, out_channels=out_channels,
                        out_kernel_initializer=tf_model_config["backbone"]["out_kernel_initializer"],
                        out_activation=tf_model_config["backbone"]["out_activation"],
                        **tf_model_config["global"])
    elif tf_model_config["backbone"]["name"] == "unet":
        return UNet(image_size=image_size, out_channels=out_channels,
                    out_kernel_initializer=tf_model_config["backbone"]["out_kernel_initializer"],
                    out_activation=tf_model_config["backbone"]["out_activation"],
                    **tf_model_config["unet"])
    else:
        raise ValueError("Unknown model name")


def build_inputs(moving_image_size: tuple, fixed_image_size: tuple, index_size: int, batch_size: int,
                 labeled: bool) -> [tf.keras.Input, tf.keras.Input, tf.keras.Input, tf.keras.Input]:
    """
    Configure a pair of moving and fixed images and a pair of moving and fixed labels as model input
    and returns model input tf.keras.Input

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param batch_size: mini-batch size
    :param labeled: true if we have label data
    :return: tf.keras.Input objects
    """
    moving_image = tf.keras.Input(shape=(*moving_image_size,), batch_size=batch_size,
                                  name="moving_image")  # (batch, m_dim1, m_dim2, m_dim3)
    fixed_image = tf.keras.Input(shape=(*fixed_image_size,), batch_size=batch_size,
                                 name="fixed_image")  # (batch, f_dim1, f_dim2, f_dim3)
    moving_label = tf.keras.Input(shape=(*moving_image_size,), batch_size=batch_size,
                                  name="moving_label") if labeled else None  # (batch, m_dim1, m_dim2, m_dim3)
    indices = tf.keras.Input(shape=(index_size,), batch_size=batch_size,
                             name="indices")  # (batch, 2)
    return moving_image, fixed_image, moving_label, indices
