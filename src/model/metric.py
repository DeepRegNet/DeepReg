import tensorflow as tf


class Metrics:
    def __init__(self, tb_names):
        """
        :param tb_names: a dict which key, value = variable name, tensorboard display name
        """
        self.metrics = dict()
        self.tb_names = tb_names
        for var_name in tb_names:
            self.metrics[var_name] = tf.keras.metrics.Mean(name=var_name)

    def update(self, metric_value_dict):
        for var_name in self.metrics:
            self.metrics[var_name](metric_value_dict[var_name])

    def update_tensorboard(self, step):
        for var_name in self.metrics:
            tf.summary.scalar(self.tb_names[var_name], self.metrics[var_name].result(), step=step)

    def reset(self):
        for var_name in self.metrics:
            self.metrics[var_name].reset_states()

    def __repr__(self):
        value_dict = dict()
        for var_name in self.metrics:
            value_dict[var_name] = self.metrics[var_name].result().numpy()
        return value_dict.__repr__()
