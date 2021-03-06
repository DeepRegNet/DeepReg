# coding=utf-8

"""
Tests for deepreg/model/backbone/u_net.py
"""
from typing import Tuple

import pytest
import tensorflow as tf

from deepreg.model.backbone.u_net import UNet


class TestUNet:
    """
    Test the backbone.u_net.UNet class
    """

    @pytest.mark.parametrize(
        "depth,encode_num_channels,decode_num_channels",
        [
            (2, (4, 8, 16), (4, 8, 16)),
            (2, (4, 8, 8), (4, 8, 8)),
            (2, (4, 8, 8), (8, 8, 8)),
        ],
    )
    @pytest.mark.parametrize("pooling", [True, False])
    @pytest.mark.parametrize("concat_skip", [True, False])
    def test_channels(
        self,
        depth: int,
        encode_num_channels: Tuple,
        decode_num_channels: Tuple,
        pooling: bool,
        concat_skip: bool,
    ):
        """
        Test unet with custom encode/decode channels.

        :param depth: input is at level 0, bottom is at level depth
        :param encode_num_channels: filters/channels for down-sampling,
            by default it is doubled at each layer during down-sampling
        :param decode_num_channels: filters/channels for up-sampling,
            by default it is the same as encode_num_channels
        :param pooling: for down-sampling, use non-parameterized
                        pooling if true, otherwise use conv3d
        :param concat_skip: if concatenate skip or add it
        """
        # in case of adding skip tensors, the channels should match
        expect_err = (not concat_skip) and encode_num_channels != decode_num_channels

        image_size = (5, 6, 7)
        out_ch = 3
        try:
            network = UNet(
                image_size=image_size,
                out_channels=out_ch,
                num_channel_initial=0,
                encode_num_channels=encode_num_channels,
                decode_num_channels=decode_num_channels,
                depth=depth,
                out_kernel_initializer="he_normal",
                out_activation="softmax",
                pooling=pooling,
                concat_skip=concat_skip,
            )
        except ValueError as err:
            if expect_err:
                return
            raise err
        inputs = tf.ones(shape=(5, *image_size, out_ch))

        output = network.call(inputs)
        assert inputs.shape == output.shape

    @pytest.mark.parametrize(
        "image_size,depth",
        [((11, 12, 13), 5), ((8, 8, 8), 3)],
    )
    @pytest.mark.parametrize("pooling", [True, False])
    @pytest.mark.parametrize("concat_skip", [True, False])
    def test_call(
        self,
        image_size: Tuple,
        depth: int,
        pooling: bool,
        concat_skip: bool,
    ):
        """
        Test unet call function.

        :param image_size: (dim1, dim2, dim3), dims of input image.
        :param depth: input is at level 0, bottom is at level depth
        :param pooling: for down-sampling, use non-parameterized
                        pooling if true, otherwise use conv3d
        :param concat_skip: if concatenate skip or add it
        """
        out_ch = 3
        network = UNet(
            image_size=image_size,
            out_channels=out_ch,
            num_channel_initial=2,
            depth=depth,
            out_kernel_initializer="he_normal",
            out_activation="softmax",
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
            encode_kernel_sizes=3,
            decode_kernel_sizes=3,
            encode_num_channels=(2, 4, 8),
            decode_num_channels=(2, 4, 8),
            strides=2,
            padding="same",
            name="Test",
        )
        network = UNet(**config)
        got = network.get_config()
        assert got == config
