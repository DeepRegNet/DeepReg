# coding=utf-8

"""
Tests for deepreg/model/backbone/u_net.py
"""
import pytest
import tensorflow as tf

from deepreg.model.backbone.u_net import UNet


class TestUNet:
    """
    Test the backbone.u_net.UNet class
    """

    @pytest.mark.parametrize(
        "image_size,depth",
        [((11, 12, 13), 5), ((8, 8, 8), 3)],
    )
    @pytest.mark.parametrize("pooling", [True, False])
    @pytest.mark.parametrize("concat_skip", [True, False])
    def test_call(
        self,
        image_size: tuple,
        depth: int,
        pooling: bool,
        concat_skip: bool,
    ):
        """

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
            strides=2,
            padding="same",
            name="Test",
        )
        network = UNet(**config)
        got = network.get_config()
        assert got == config
