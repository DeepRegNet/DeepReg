# coding=utf-8

"""
Tests for deepreg/model/optimizer.py
pytest style
"""
import tensorflow as tf

import deepreg.model.optimizer as optimizer


class TestBuildOptimizer:
    def test_build_optimizer_adam(self):
        """Build an Adam optimizer"""
        opt_config = {"name": "Adam", "learning_rate": 1.0e-5}
        opt_get = optimizer.build_optimizer(opt_config)
        assert isinstance(opt_get, tf.keras.optimizers.Adam)

    def test_build_optimizer_sgd(self):
        """Build an SGD optimizer"""
        opt_config = {"name": "SGD"}
        opt_get = optimizer.build_optimizer(opt_config)
        assert isinstance(opt_get, tf.keras.optimizers.SGD)
