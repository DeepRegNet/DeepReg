'''
Functions parsing the config optimiser options
:function 
'''

import tensorflow as tf


def get_optimizer(tf_opt_config):
    '''
    Parsing the optimiser options and parameters

    :param tf_opt_config: unpacked key-value pairs for the optimiser returned from yaml.load, optimiser options and parameters

    :return: tf.keras.optimizers object

    '''
    if tf_opt_config["name"] == "adam":
        return tf.keras.optimizers.Adam(**tf_opt_config["adam"])
    elif tf_opt_config["name"] == "sgd":
        return tf.keras.optimizers.SGD(**tf_opt_config["sgd"])
    elif tf_opt_config["name"] == "rms":
        return tf.keras.optimizers.RMSprop(**tf_opt_config["rms"])
    else:
        raise ValueError("Unknown optimizer")
