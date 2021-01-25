"""
Functions parsing the config optimizer options
"""

import tensorflow as tf


def build_optimizer(optimizer_config: dict) -> tf.optimizers.Optimizer:
    """
    Parsing the optimiser options and parameters
    from config dictionary.

    :param optimizer_config: has key name and other required arguments
    :return: optimizer instant
    """

    optimizer_cls = getattr(tf.keras.optimizers, optimizer_config["name"])
    optimizer = optimizer_cls(**optimizer_config)
    return optimizer
