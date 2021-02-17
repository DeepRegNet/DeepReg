from abc import abstractmethod

import tensorflow as tf


class Backbone(tf.keras.Model):
    """
    Interface class for backbones.
    """

    def __init__(
        self,
        image_size: tuple,
        num_channel_initial: int,
        out_kernel_initializer: str,
        out_activation: str,
        out_channels: int,
        name: str = "Backbone",
        **kwargs,
    ):
        """
        Init.

        :param image_size: (dim1, dim2, dim3), dims of input image.
        :param num_channel_initial: number of initial channels, control the network size
        :param out_kernel_initializer: kernel initializer for the last layer
        :param out_activation: activation at the last layer
        :param out_channels: number of channels for the output
        :param name: name of the backbone.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)

        self.image_size = image_size
        self.num_channel_initial = num_channel_initial
        self.out_kernel_initializer = out_kernel_initializer
        self.out_activation = out_activation
        self.out_channels = out_channels

    @abstractmethod
    def call(self, inputs: tf.Tensor, training=None, mask=None):
        """
        Forward.

        :param inputs: shape = (batch, dim1, dim2, dim3, in_channels)
        :param training:
        :param mask:
        :return:
        """

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        return dict(
            image_size=self.image_size,
            num_channel_initial=self.num_channel_initial,
            out_kernel_initializer=self.out_kernel_initializer,
            out_activation=self.out_activation,
            out_channels=self.out_channels,
            name=self.name,
        )
