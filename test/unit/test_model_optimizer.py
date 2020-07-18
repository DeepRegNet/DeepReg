# coding=utf-8

"""
Tests for deepreg/model/optimizer.py
pytest style
"""
import pytest
import tensorflow

import deepreg.model.optimizer as optimizer


def test_get_optimizer_not_dict():
    """
    Test assertion error raised if
    config passed not dict.
    """
    with pytest.raises(AssertionError):
        optimizer.build_optimizer(["name"])


def test_get_optimizer_adam():
    """Assert that correct keras optimizer
    is returned when passing the adam string
    into get_optimizer function
    """
    dict_config = {"name": "adam", "adam": {}}
    opt_get = optimizer.build_optimizer(dict_config)
    assert isinstance(opt_get, tensorflow.python.keras.optimizer_v2.adam.Adam)


def test_get_optimizer_sgd():
    """
    Assert that correct keras optimizer
    is returned when passing the sgd string
    into get_optimizer function
    """
    dict_config = {"name": "sgd", "sgd": {}}
    opt_get = optimizer.build_optimizer(dict_config)
    assert isinstance(
        opt_get, tensorflow.python.keras.optimizer_v2.gradient_descent.SGD
    )


def test_get_optimizer_rms():
    """
    Assert that correct keras optimizer
    is returned when passing the rms string
    into get_optimizer function
    """
    dict_config = {"name": "rms", "rms": {}}
    opt_get = optimizer.build_optimizer(dict_config)
    assert isinstance(opt_get, tensorflow.python.keras.optimizer_v2.rmsprop.RMSprop)


def test_get_optimizer_error():
    """
    Assert value_error raised if
    unknown optimizer type is passed
    to get_optimizer func,
    """
    with pytest.raises(ValueError):
        optimizer.build_optimizer({"name": "random"})
