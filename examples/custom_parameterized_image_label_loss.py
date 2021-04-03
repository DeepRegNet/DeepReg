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
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "LPNorm",
    ):
        """
        Init.

        :param p: order of the norm, 1 or 2.
        :param reduction: using SUM reduction over batch axis,
            this is for supporting multi-device training,
            and the loss will be divided by global batch size,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss.
        """
        super().__init__(reduction=reduction, name=name)
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
