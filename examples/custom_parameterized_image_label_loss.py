"""This script provides an example of using custom backbone for training."""
import tensorflow as tf

from deepreg.registry import REGISTRY
from deepreg.train import train


@REGISTRY.register_loss(name="lp_norm")
class LPNorm(tf.keras.losses.Loss):
    """
    L^p norm between y_true and y_pred, p = 1 or 2.

    y_true and y_pred have to be at least 1d tensor, including batch axis.
    """

    def __init__(
        self,
        p: int,
        name: str = "LPNorm",
        **kwargs,
    ):
        """
        Init.

        :param p: order of the norm, 1 or 2.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        if p not in [1, 2]:
            raise ValueError(f"For LPNorm, p must be 0 or 1, got {p}.")
        self.p = p
        self.flatten = tf.keras.layers.Flatten()

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        diff = y_true - y_pred
        diff = self.flatten(diff)
        loss = tf.norm(diff, axis=-1, ord=self.p)
        return loss


config_path = "examples/config_custom_parameterized_image_label_loss.yaml"
train(
    gpu="",
    config_path=config_path,
    gpu_allow_growth=True,
    ckpt_path="",
)
