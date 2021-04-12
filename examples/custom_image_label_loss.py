"""This script provides an example of using custom backbone for training."""
import tensorflow as tf

from deepreg.registry import REGISTRY
from deepreg.train import train


@REGISTRY.register_loss(name="root_mean_square")
class RootMeanSquaredDifference(tf.keras.losses.Loss):
    """
    Square root of the mean of squared distance between y_true and y_pred.

    y_true and y_pred have to be at least 1d tensor, including batch axis.
    """

    def __init__(
        self,
        name: str = "RootMeanSquaredDifference",
        **kwargs,
    ):
        """
        Init.

        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        loss = tf.math.squared_difference(y_true, y_pred)
        loss = self.flatten(loss)
        loss = tf.reduce_mean(loss, axis=1)
        loss = tf.math.sqrt(loss)
        return loss


config_path = "examples/config_custom_image_label_loss.yaml"
train(
    gpu="",
    config_path=config_path,
    gpu_allow_growth=True,
    ckpt_path="",
)
