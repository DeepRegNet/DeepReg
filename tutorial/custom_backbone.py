import tensorflow as tf

from deepreg.registry import REGISTRY
from deepreg.train import train


@REGISTRY.register_backbone(name="custom")
class CustomBackbone(tf.keras.Model):
    """
    A dummy custom model for demonstration purpose only
    """

    def __init__(
        self,
        image_size: tuple,
        out_channels: int,
        num_channel_initial: int,
        out_kernel_initializer: str,
        out_activation: str,
        **kwargs,
    ):
        """
        Init.

        :param image_size: (dim1, dim2, dim3), dims of input image.
        :param out_channels: number of channels for the output
        :param num_channel_initial: number of initial channels
        :param depth: input is at level 0, bottom is at level depth
        :param out_kernel_initializer: which kernel to use as initializer
        :param out_activation: activation at last layer
        :param kwargs: additional arguments.
        """
        super().__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv3D(
            filters=num_channel_initial, kernel_size=3, padding="same"
        )
        self.conv2 = tf.keras.layers.Conv3D(
            filters=out_channels,
            kernel_size=1,
            kernel_initializer=out_kernel_initializer,
            activation=out_activation,
            padding="same",
        )

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> tf.Tensor:
        """
        Builds graph based on built layers.

        :param inputs: shape = [batch, f_dim1, f_dim2, f_dim3, in_channels]
        :param training:
        :param mask:
        :return: shape = [batch, f_dim1, f_dim2, f_dim3, out_channels]
        """
        out = self.conv1(inputs)
        out = self.conv2(out)
        return out


config_path = "tutorial/config_custom_backbone.yaml"
train(
    gpu="",
    config_path=config_path,
    gpu_allow_growth=True,
    ckpt_path="",
)
