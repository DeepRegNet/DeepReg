import tensorflow as tf


class Metrics:
    def __init__(self, metric_names):
        self.metrics = dict()
        for metric_name in metric_names:
            self.metrics[metric_name] = tf.keras.metrics.Mean(name=metric_name)

    def update(self, metric_value_dict):
        for metric_name in self.metrics:
            self.metrics[metric_name](metric_value_dict[metric_name])

    def update_tensorboard(self, step):
        for metric_name in self.metrics:
            tf.summary.scalar(metric_name, self.metrics[metric_name].result(), step=step)

    def reset(self):
        for metric_name in self.metrics:
            self.metrics[metric_name].reset_states()

    def __repr__(self):
        value_dict = dict()
        for metric_name in self.metrics:
            value_dict[metric_name] = self.metrics[metric_name].result().numpy()
        return value_dict.__repr__()
