import tensorflow as tf


class NegativeLossMixin(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super(NegativeLossMixin, self).__init__(**kwargs)
        self.name = self.name + "Loss"

    def call(self, y_true, y_pred):
        return -super(NegativeLossMixin, self).call(y_true=y_true, y_pred=y_pred)
