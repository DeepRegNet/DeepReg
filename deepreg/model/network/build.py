from deepreg.model.network.cond import build_cond_model
from deepreg.model.network.ddf import build_ddf_model
from deepreg.model.network.dvf import build_dvf_model


def build_model(moving_image_size: tuple, fixed_image_size: tuple, index_size: int, labeled: bool, batch_size: int,
                tf_model_config: dict, tf_loss_config: dict):
    """
    Parsing algorithm types to model building functions

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param labeled: true if the label of moving/fixed images are provided
    :param batch_size: mini-batch size
    :param tf_model_config: model configuration, e.g. dictionary return from parser.yaml.load
    :param tf_loss_config: loss configuration, e.g. dictionary return from parser.yaml.load
    :return: the built tf.keras.Model
    """
    if tf_model_config["method"] == "ddf":
        return build_ddf_model(moving_image_size, fixed_image_size, index_size, labeled, batch_size, tf_model_config,
                               tf_loss_config)
    elif tf_model_config["method"] == "dvf":
        return build_dvf_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config,
                               tf_loss_config)
    elif tf_model_config["method"] == "conditional":
        return build_cond_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config,
                                tf_loss_config)
    else:
        raise ValueError("Unknown model method")
