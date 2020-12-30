"""Provide helper functions or classes for defining loss or metrics."""
import tensorflow as tf


class NegativeLossMixin(tf.keras.losses.Loss):
    """Mixin class to revert the sign of the loss value."""

    def __init__(self, **kwargs):
        """Init."""
        super().__init__(**kwargs)
        self.name = self.name + "Loss"

    def call(self, y_true, y_pred):
        """Revert the sign of loss."""
        return -super().call(y_true=y_true, y_pred=y_pred)
