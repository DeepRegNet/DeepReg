import tensorflow as tf


def get_optimizer(tf_opt_config):
    if tf_opt_config["name"] == "adam":
        return tf.keras.optimizers.Adam(**tf_opt_config["adam"])
    elif tf_opt_config["name"] == "sgd":
        return tf.keras.optimizers.SGD(**tf_opt_config["sgd"])
    elif tf_opt_config["name"] == "rms":
        return tf.keras.optimizers.RMSprop(**tf_opt_config["rms"])
    else:
        raise ValueError("Unknown optimizer")
