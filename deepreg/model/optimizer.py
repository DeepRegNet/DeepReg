"""
Functions parsing the config optimizer options
"""

import tensorflow as tf


def build_optimizer(optimizer_config: dict):
    """
    Parsing the optimiser options and parameters
    from config dictionary.

    :param optimizer_config: unpacked dictionary for
      the optimiser returned from yaml.load, optimiser options
      and parameters
    :return: tf.keras.optimizers object
    """
    assert isinstance(optimizer_config, dict)

    if optimizer_config["name"] == "adam":
        return tf.keras.optimizers.Adam(**optimizer_config["adam"])
    elif optimizer_config["name"] == "sgd":
        return tf.keras.optimizers.SGD(**optimizer_config["sgd"])
    elif optimizer_config["name"] == "rms":
        return tf.keras.optimizers.RMSprop(**optimizer_config["rms"])
    else:
        raise ValueError("Unknown optimizer")
