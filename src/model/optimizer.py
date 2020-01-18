import tensorflow as tf


def get_optimizer(tf_opt_config):
    if tf_opt_config["name"] == "adam":
        return tf.keras.optimizers.Adam(**tf_opt_config["adam"])
    else:
        raise ValueError("Unknown optimizer")
