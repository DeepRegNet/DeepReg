from deepreg.model.network.affine import build_affine_model
from deepreg.model.network.cond import build_conditional_model
from deepreg.model.network.ddf_dvf import build_ddf_dvf_model


def build_model(
    moving_image_size: tuple,
    fixed_image_size: tuple,
    index_size: int,
    labeled: bool,
    batch_size: int,
    model_config: dict,
    loss_config: dict,
):
    """
    Parsing algorithm types to model building functions.

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param labeled: true if the label of moving/fixed images are provided
    :param batch_size: mini-batch size
    :param model_config: model configuration, e.g. dictionary return from parser.yaml.load
    :param loss_config: loss configuration, e.g. dictionary return from parser.yaml.load
    :return: the built tf.keras.Model
    """
    if model_config["method"] in ["ddf", "dvf"]:
        return build_ddf_dvf_model(
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            index_size=index_size,
            labeled=labeled,
            batch_size=batch_size,
            model_config=model_config,
            loss_config=loss_config,
        )
    elif model_config["method"] == "conditional":
        return build_conditional_model(
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            index_size=index_size,
            labeled=labeled,
            batch_size=batch_size,
            model_config=model_config,
            loss_config=loss_config,
        )
    elif model_config["method"] == "affine":
        return build_affine_model(
            moving_image_size=moving_image_size,
            fixed_image_size=fixed_image_size,
            index_size=index_size,
            labeled=labeled,
            batch_size=batch_size,
            model_config=model_config,
            loss_config=loss_config,
        )
    else:
        raise ValueError("Unknown model method")
