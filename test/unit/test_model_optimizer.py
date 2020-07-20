# coding=utf-8

"""
Tests for deepreg/model/optimizer.py
pytest style
"""
import pytest
import tensorflow as tf

import deepreg.model.optimizer as optimizer


def test_get_optimizer_not_dict():
    """
    Test assertion error raised if
    config passed not dict.
    """
    with pytest.raises(AssertionError):
        optimizer.get_optimizer(["name"])


def test_get_optimizer_adam():
    """Assert that correct keras optimizer
    is returned when passing the adam string
    into get_optimizer function
    """
    dict_config = {"name": "adam", "adam": {}}
    opt_get = optimizer.get_optimizer(dict_config)
    assert isinstance(opt_get, tf.keras.optimizers.Adam)


def test_get_optimizer_sgd():
    """
    Assert that correct keras optimizer
    is returned when passing the sgd string
    into get_optimizer function
    """
    dict_config = {"name": "sgd", "sgd": {}}
    opt_get = optimizer.get_optimizer(dict_config)
    assert isinstance(opt_get, tf.keras.optimizers.SGD)


def test_get_optimizer_rms():
    """
    Assert that correct keras optimizer
    is returned when passing the rms string
    into get_optimizer function
    """
    dict_config = {"name": "rms", "rms": {}}
    opt_get = optimizer.get_optimizer(dict_config)
    assert isinstance(opt_get, tf.keras.optimizers.RMSprop)


def test_get_optimizer_error():
    """
    Assert value_error raised if
    unknown optimizer type is passed
    to get_optimizer func,
    """
    with pytest.raises(ValueError):
        optimizer.get_optimizer({"name": "random"})
