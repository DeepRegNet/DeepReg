# coding=utf-8

"""
Tests for deepreg/model/backbone/local_net.py
"""
from typing import Tuple

import pytest
import tensorflow as tf

from deepreg.model.backbone.local_net import AdditiveUpsampling, LocalNet


def test_additive_up_sampling():
    """
    Test AdditiveUpsampling.
    """
    batch = 3
    filters = 4
    input_shape = (4, 5, 6)
    outputs_shape = tuple(x * 2 for x in input_shape)
    config = dict(
        filters=filters,
        output_padding=(1, 1, 1),
        kernel_size=3,
        padding="same",
        strides=2,
        output_shape=outputs_shape,
        name="TestAdditiveUpsampling",
    )
    layer = AdditiveUpsampling(**config)
    inputs = tf.ones(shape=(batch, *input_shape, filters * 2))
    output = layer.call(inputs)
    assert output.shape == (batch, *outputs_shape, filters)

    got = layer.get_config()
    assert got == {"trainable": True, "dtype": "float32", **config}


class TestLocalNet:
    """
    Test the backbone.local_net.LocalNet class
    """

    @pytest.mark.parametrize(
        "image_size,extract_levels,depth",
        [((11, 12, 13), (0, 1, 2, 4), 4), ((8, 8, 8), (0, 1, 2), 3)],
    )
    @pytest.mark.parametrize("use_additive_upsampling", [True, False])
    @pytest.mark.parametrize("pooling", [True, False])
    @pytest.mark.parametrize("concat_skip", [True, False])
    def test_call(
        self,
        image_size: tuple,
        extract_levels: Tuple[int, ...],
        depth: int,
        use_additive_upsampling: bool,
        pooling: bool,
        concat_skip: bool,
    ):
        """

        :param image_size: (dim1, dim2, dim3), dims of input image.
        :param extract_levels: from which depths the output will be built.
        :param depth: input is at level 0, bottom is at level depth
        :param use_additive_upsampling: whether use additive up-sampling layer
            for decoding.
        :param pooling: for down-sampling, use non-parameterized
                        pooling if true, otherwise use conv3d
        :param concat_skip: if concatenate skip or add it
        """
        out_ch = 3
        network = LocalNet(
            image_size=image_size,
            num_channel_initial=2,
            extract_levels=extract_levels,
            depth=depth,
            out_kernel_initializer="he_normal",
            out_activation="softmax",
            out_channels=out_ch,
            use_additive_upsampling=use_additive_upsampling,
            pooling=pooling,
            concat_skip=concat_skip,
        )
        inputs = tf.ones(shape=(5, *image_size, out_ch))
        output = network.call(inputs)
        assert inputs.shape == output.shape

    def test_get_config(self):
        config = dict(
            image_size=(4, 5, 6),
            out_channels=3,
            num_channel_initial=2,
            depth=2,
            extract_levels=(0, 1),
            out_kernel_initializer="he_normal",
            out_activation="softmax",
            pooling=False,
            concat_skip=False,
            use_additive_upsampling=True,
            encode_kernel_sizes=[7, 3, 3],
            decode_kernel_sizes=3,
            encode_num_channels=(2, 4, 8),
            decode_num_channels=(2, 4, 8),
            strides=2,
            padding="same",
            name="Test",
        )
        network = LocalNet(**config)
        got = network.get_config()
        assert got == config
