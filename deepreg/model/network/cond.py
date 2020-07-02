import tensorflow as tf

from deepreg.model import layer as layer
from deepreg.model.network.util import build_backbone, build_inputs


def build_cond_model(
    moving_image_size,
    fixed_image_size,
    index_size,
    batch_size,
    model_config,
    loss_config,
):
    """
    Build the model if using conditional segmentation algorithm

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param batch_size: mini-batch size
    :param model_config: model configuration, e.g. dictionary return from parser.yaml.load
    :param loss_config: loss configuration, e.g. dictionary return from parser.yaml.load
    :return: the built tf.keras.Model
    """
    # inputs
    moving_image, fixed_image, moving_label, indices = build_inputs(
        moving_image_size, fixed_image_size, index_size, batch_size
    )
    backbone_input = tf.concat(
        [
            layer.Resize3d(size=fixed_image_size)(
                inputs=tf.expand_dims(moving_image, axis=4)
            ),
            tf.expand_dims(fixed_image, axis=4),
            layer.Resize3d(size=fixed_image_size)(
                inputs=tf.expand_dims(moving_label, axis=4)
            ),
        ],
        axis=4,
    )  # [batch, f_dim1, f_dim2, f_dim3, 3]

    # backbone
    backbone = build_backbone(
        image_size=fixed_image_size, out_channels=1, tf_model_config=model_config
    )

    # prediction
    pred_fixed_label = backbone(
        inputs=backbone_input
    )  # [batch, f_dim1, f_dim2, f_dim3, 1]
    pred_fixed_label = tf.squeeze(pred_fixed_label, axis=4)

    # build model
    model = tf.keras.Model(
        inputs=[moving_image, fixed_image, moving_label, indices],
        outputs=[pred_fixed_label],
        name="CondRegModel",
    )

    return model
