import tensorflow as tf


def get_optimizer(tf_opt_config):
    if tf_opt_config["name"] == "adam":
        return tf.keras.optimizers.Adam(**tf_opt_config["adam"])
    elif tf_opt_config["name"] == "sgd":
        return tf.keras.optimizers.SGD(**tf_opt_config["sgd"])
    else:
        raise ValueError("Unknown optimizer")
