from unittest import TestCase

import numpy as np
import tensorflow as tf

import deepreg.model.layer_util as layer_util


class Test(TestCase):
    @staticmethod
    def check_equal(x, y):
        """
        given two tf tensors return True/False (not tf tensor)
        tolerate small errors
        :param x:
        :param y:
        :return:
        """
        return tf.reduce_max(tf.abs(x - y)).numpy() < 1e-6

    def test_get_reference_grid(self):
        want = tf.constant(np.array(
            [[[[0, 0, 0],
               [0, 0, 1],
               [0, 0, 2]],
              [[0, 1, 0],
               [0, 1, 1],
               [0, 1, 2]]]]
            , dtype=np.float32))
        get = layer_util.get_reference_grid(grid_size=[1, 2, 3])
        self.check_equal(want, get)

    def test_resample(self):
        # linear, vol has no feature channel
        interpolation = "linear"
        vol = tf.constant(np.array(
            [[[0, 1, 2],
              [3, 4, 5],
              ]],
            dtype=np.float32))  # shape = [1,2,3]
        loc = tf.constant(np.array(
            [[[[0, 0],
               [0, 1],
               [0, 3]],  # outside frame
              [[0.4, 0],
               [0.5, 1],
               [0.6, 2]],
              [[0.4, 0.7],
               [0.5, 0.5],
               [0.6, 0.3]],  # resampled = 3x+y
              ]],
            dtype=np.float32))  # shape = [1,3,3,2]
        want = tf.constant(np.array(
            [[[0, 1, 2],
              [1.2, 2.5, 3.8],
              [1.9, 2, 2.1],
              ]],
            dtype=np.float32))  # shape = [1,3,3]
        get = layer_util.resample(vol=vol, loc=loc, interpolation=interpolation)
        self.check_equal(want, get)

        # linear, vol has feature channel
        interpolation = "linear"
        vol = tf.constant(np.array(
            [[[[0, 0],
               [1, 1],
               [2, 2],
               ],
              [[3, 3],
               [4, 4],
               [5, 5],
               ],
              ]],
            dtype=np.float32))  # shape = [1,2,3,2]
        loc = tf.constant(np.array(
            [[[[0, 0],
               [0, 1],
               [0, 3]],  # outside frame
              [[0.4, 0],
               [0.5, 1],
               [0.6, 2]],
              [[0.4, 0.7],
               [0.5, 0.5],
               [0.6, 0.3]],  # resampled = 3x+y
              ]],
            dtype=np.float32))  # shape = [1,3,3,2]
        want = tf.constant(np.array(
            [[[[0, 0],
               [1, 1],
               [2, 2]],
              [[1.2, 1.2],
               [2.5, 2.5],
               [3.8, 3.8]],
              [[1.9, 1.9],
               [2, 2],
               [2.1, 2.1]],
              ]],
            dtype=np.float32))  # shape = [1,3,3,2]
        get = layer_util.resample(vol=vol, loc=loc, interpolation=interpolation)
        self.check_equal(want, get)
