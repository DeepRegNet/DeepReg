import tensorflow as tf


def get_optimizer(config_model):
    if config_model["opt"]["optimizer"] == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=config_model["opt"]["learning_rate"],
                                             clipnorm=config_model["opt"]["clipnorm"])
    else:
        raise ValueError("Unknown optimizer %s." % config_model["opt"]["optimizer"])
    return optimizer
