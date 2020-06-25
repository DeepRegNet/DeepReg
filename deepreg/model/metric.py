import tensorflow as tf

import deepreg.model.layer_util as layer_util
import deepreg.model.loss.label as label_loss

EPS = 1.0e-6  # epsilon to prevent NaN


class MeanWrapper(tf.keras.metrics.Metric):
    def __init__(self, name, **kwargs):
        super(MeanWrapper, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def fn(self, y_true, y_pred):
        # return values of size [batch]
        raise NotImplementedError

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = self.fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(values))
        self.count.assign_add(tf.reduce_sum(tf.ones_like(values)))

    def result(self):
        return (self.total + EPS) / (self.count + EPS)

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.total.assign(0.0)
        self.count.assign(0.0)


class MeanDiceScore(MeanWrapper):
    def __init__(self, name="metric/binary_dice_score_mean", **kwargs):
        super(MeanDiceScore, self).__init__(name=name, **kwargs)

    def fn(self, y_true, y_pred):
        return label_loss.dice_score(y_true=y_true, y_pred=y_pred, binary=True)


class MeanCentroidDistance(MeanWrapper):
    def __init__(
        self, grid_size, name="metric/centroid_distance_mean", **kwargs
    ):
        super(MeanCentroidDistance, self).__init__(name=name, **kwargs)
        self.grid = layer_util.get_reference_grid(grid_size)

    def fn(self, y_true, y_pred):
        return label_loss.compute_centroid_distance(y_true, y_pred, self.grid)


class MeanForegroundProportion(MeanWrapper):
    def __init__(
        self, pred: bool, name="metric/foreground_proportion", **kwargs
    ):
        name += "_pred" if pred else "_true"
        super(MeanForegroundProportion, self).__init__(name=name, **kwargs)
        self.pred = pred

    def fn(self, y_true, y_pred):
        y = y_pred if self.pred else y_true
        y = tf.cast(y >= 0.5, dtype=tf.float32)
        return tf.reduce_sum(y, axis=[1, 2, 3]) / tf.reduce_sum(
            tf.ones_like(y), axis=[1, 2, 3]
        )
