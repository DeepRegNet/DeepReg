# coding=utf-8

"""
Tests for deepreg/model/optimizer.py
pytest style
"""
import pytest
import tensorflow as tf

import deepreg.model.optimizer as optimizer


def test_build_optimizer_not_dict():
    """
    Test assertion error raised if
    config passed not dict.
    """
    with pytest.raises(AssertionError):
        optimizer.build_optimizer(["name"])


def test_build_optimizer_adam():
    """Assert that correct keras optimizer
    is returned when passing the adam string
    into build_optimizer function
    """
    dict_config = {"name": "adam", "adam": {}}
    opt_get = optimizer.build_optimizer(dict_config)
    assert isinstance(opt_get, tf.keras.optimizers.Adam)


def test_build_optimizer_sgd():
    """
    Assert that correct keras optimizer
    is returned when passing the sgd string
    into build_optimizer function
    """
    dict_config = {"name": "sgd", "sgd": {}}
    opt_get = optimizer.build_optimizer(dict_config)
    assert isinstance(opt_get, tf.keras.optimizers.SGD)


def test_build_optimizer_rms():
    """
    Assert that correct keras optimizer
    is returned when passing the rms string
    into build_optimizer function
    """
    dict_config = {"name": "rms", "rms": {}}
    opt_get = optimizer.build_optimizer(dict_config)
    assert isinstance(opt_get, tf.keras.optimizers.RMSprop)


def test_build_optimizer_error():
    """
    Assert value_error raised if
    unknown optimizer type is passed
    to build_optimizer func,
    """
    with pytest.raises(ValueError):
        optimizer.build_optimizer({"name": "random"})
