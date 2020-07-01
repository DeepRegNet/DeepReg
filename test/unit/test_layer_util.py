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
        )  # shape = (1, 2, 3, 3)
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
            [0, 0, 0],  # 0
            [0, 0, 1],  # 1
            [0, 1, 0],  # 2
            [0, 1, 1],  # 3
            [1, 0, 0],  # 4
            [1, 0, 1],  # 5
            [1, 1, 0],  # 6
            [1, 1, 1],  # 7
        ]
        got = layer_util.get_n_bits_combinations(3)
        self.assertEqual(got, expected)

    def test_pyramid_combination(self):
        # num_dim = 1
        weights = tf.constant(np.array([[0.2]], dtype=np.float32))
        values = tf.constant(np.array([[1], [2]], dtype=np.float32))
        # expected = 1 * 0.2 + 2 * 2
        expected = tf.constant(np.array([1.8], dtype=np.float32))
        got = layer_util.pyramid_combination(values=values, weights=weights)
        self.assertTensorsEqual(got, expected)

        # num_dim = 2
        weights = tf.constant(np.array([[0.2], [0.3]], dtype=np.float32))
        values = tf.constant(
            np.array(
                [
                    [1],  # value at corner (0, 0), weight = 0.2 * 0.3
                    [2],  # value at corner (0, 1), weight = 0.2 * 0.7
                    [3],  # value at corner (1, 0), weight = 0.8 * 0.3
                    [4],  # value at corner (1, 1), weight = 0.8 * 0.7
                ],
                dtype=np.float32,
            )
        )
        # expected = 1 * 0.2 * 0.3
        #          + 2 * 0.2 * 0.7
        #          + 3 * 0.8 * 0.3
        #          + 4 * 0.8 * 0.7
        expected = tf.constant(np.array([3.3], dtype=np.float32))
        got = layer_util.pyramid_combination(values=values, weights=weights)
        self.assertTensorsEqual(got, expected)

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

    def test_random_transform_generator(self):
        batch_size = 1
        scale = 0.1
        seed = 0
        expected = tf.constant(
            np.array(
                [
                    [
                        [9.4661278e-01, -3.8267835e-03, 3.6934228e-03],
                        [5.5613145e-03, 9.8034811e-01, -1.8044969e-02],
                        [1.9651605e-04, 1.4576728e-02, 9.6243286e-01],
                        [-2.5107686e-03, 1.9579126e-02, -1.2195010e-02],
                    ]
                ],
                dtype=np.float32,
            )
        )  # shape = (1, 4, 3)
        got = layer_util.random_transform_generator(
            batch_size=batch_size, scale=scale, seed=seed
        )
        self.assertTensorsEqual(got, expected)

    def test_warp_grid(self):
        grid = tf.constant(
            np.array(
                [
                    [
                        [[0, 0, 0], [0, 0, 1], [0, 0, 2]],
                        [[0, 1, 0], [0, 1, 1], [0, 1, 2]],
                    ]
                ],
                dtype=np.float32,
            )
        )  # shape = (1, 2, 3, 3)
        theta = tf.constant(
            np.array(
                [
                    [
                        [0.86, 0.75, 0.48],
                        [0.07, 0.98, 0.01],
                        [0.72, 0.52, 0.97],
                        [0.12, 0.4, 0.04],
                    ]
                ],
                dtype=np.float32,
            )
        )  # shape = (1, 4, 3)
        expected = tf.constant(
            np.array(
                [
                    [
                        [
                            [[0.12, 0.4, 0.04], [0.84, 0.92, 1.01], [1.56, 1.44, 1.98]],
                            [[0.19, 1.38, 0.05], [0.91, 1.9, 1.02], [1.63, 2.42, 1.99]],
                        ]
                    ]
                ],
                dtype=np.float32,
            )
        )  # shape = (1, 1, 2, 3, 3)
        got = layer_util.warp_grid(grid=grid, theta=theta)
        self.assertTensorsEqual(got, expected)
