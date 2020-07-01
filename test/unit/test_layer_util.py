import unittest

import numpy as np
import tensorflow as tf

import deepreg.model.layer_util as layer_util


class TestLayerUtil(unittest.TestCase):
    @staticmethod
    def assertTensorsEqual(x, y):
        """
        given two tf tensors return True/False (not tf tensor)
        tolerate small errors
        :param x:
        :param y:
        :return:
        """
        return tf.reduce_max(tf.abs(x - y)).numpy() < 1e-6

    def test_get_reference_grid(self):
        expected = tf.constant(
            np.array(
                [
                    [
                        [[0, 0, 0], [0, 0, 1], [0, 0, 2]],
                        [[0, 1, 0], [0, 1, 1], [0, 1, 2]],
                    ]
                ],
                dtype=np.float32,
            )
        )
        got = layer_util.get_reference_grid(grid_size=[1, 2, 3])
        self.assertTensorsEqual(got, expected)

    def test_get_n_bits_combinations(self):
        # num_bits = 1
        expected = [[0], [1]]
        got = layer_util.get_n_bits_combinations(1)
        self.assertEqual(got, expected)
        # num_bits = 2
        expected = [[0, 0], [0, 1], [1, 0], [1, 1]]
        got = layer_util.get_n_bits_combinations(2)
        self.assertEqual(got, expected)
        # num_bits = 3
        expected = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ]
        got = layer_util.get_n_bits_combinations(3)
        self.assertEqual(got, expected)

    def test_resample_linear_without_feature(self):
        # linear, vol has no feature channel
        interpolation = "linear"
        vol = tf.constant(
            np.array([[[0, 1, 2], [3, 4, 5]]], dtype=np.float32)
        )  # shape = [1,2,3]
        loc = tf.constant(
            np.array(
                [
                    [
                        [[0, 0], [0, 1], [0, 3]],  # outside frame
                        [[0.4, 0], [0.5, 1], [0.6, 2]],
                        [[0.4, 0.7], [0.5, 0.5], [0.6, 0.3]],  # resampled = 3x+y
                    ]
                ],
                dtype=np.float32,
            )
        )  # shape = [1,3,3,2]
        expected = tf.constant(
            np.array([[[0, 1, 2], [1.2, 2.5, 3.8], [1.9, 2, 2.1]]], dtype=np.float32)
        )  # shape = [1,3,3]
        got = layer_util.resample(vol=vol, loc=loc, interpolation=interpolation)
        self.assertTensorsEqual(got, expected)

    def test_resample_linear_with_feature(self):
        # linear, vol has feature channel
        interpolation = "linear"
        vol = tf.constant(
            np.array(
                [[[[0, 0], [1, 1], [2, 2]], [[3, 3], [4, 4], [5, 5]]]], dtype=np.float32
            )
        )  # shape = [1,2,3,2]
        loc = tf.constant(
            np.array(
                [
                    [
                        [[0, 0], [0, 1], [0, 3]],  # outside frame
                        [[0.4, 0], [0.5, 1], [0.6, 2]],
                        [[0.4, 0.7], [0.5, 0.5], [0.6, 0.3]],  # resampled = 3x+y
                    ]
                ],
                dtype=np.float32,
            )
        )  # shape = [1,3,3,2]
        expected = tf.constant(
            np.array(
                [
                    [
                        [[0, 0], [1, 1], [2, 2]],
                        [[1.2, 1.2], [2.5, 2.5], [3.8, 3.8]],
                        [[1.9, 1.9], [2, 2], [2.1, 2.1]],
                    ]
                ],
                dtype=np.float32,
            )
        )  # shape = [1,3,3,2]
        got = layer_util.resample(vol=vol, loc=loc, interpolation=interpolation)
        self.assertTensorsEqual(got, expected)
