# coding=utf-8

"""
Tests for deepreg/model/backbone/global_net.py
"""
from typing import Tuple

import pytest
import tensorflow as tf

from deepreg.model.backbone.global_net import AffineHead, GlobalNet


def test_affine_head():
    """
    Test AffineHead.
    """
    batch = 3
    input_shape = (4, 5, 6)
    config = dict(image_size=input_shape, name="TestAffineHead")
    layer = AffineHead(**config)
    inputs = tf.ones(shape=(batch, *input_shape, 2))
    ddf, theta = layer.call(inputs)
    assert ddf.shape == (batch, *input_shape, 3)
    assert theta.shape == (batch, 4, 3)

    got = layer.get_config()
    assert got == {"trainable": True, "dtype": "float32", **config}


class TestGlobalNet:
    """
    Test the backbone.global_net.GlobalNet class
    """

    @pytest.mark.parametrize(
        "image_size,extract_levels,depth",
        [
            ((11, 12, 13), (0, 1, 2, 4), 4),
            ((11, 12, 13), None, 4),
            ((11, 12, 13), (0, 1, 2, 4), None),
            ((8, 8, 8), (0, 1, 2), 3),
        ],
    )
    def test_call(
        self,
        image_size: tuple,
        extract_levels: Tuple[int, ...],
        depth: int,
    ):
        """

        :param image_size: (dim1, dim2, dim3), dims of input image.
        :param extract_levels: from which depths the output will be built.
        :param depth: input is at level 0, bottom is at level depth
        """
        batch_size = 5
        out_ch = 3
        network = GlobalNet(
            image_size=image_size,
            num_channel_initial=2,
            extract_levels=extract_levels,
            depth=depth,
            out_kernel_initializer="he_normal",
            out_activation="softmax",
            out_channels=out_ch,
        )
        inputs = tf.ones(shape=(batch_size, *image_size, out_ch))
        ddf, theta = network.call(inputs)
        assert ddf.shape == inputs.shape
        assert theta.shape == (batch_size, 4, 3)

    def test_err(self):
        with pytest.raises(ValueError) as err_info:
            GlobalNet(
                image_size=(4, 5, 6),
                out_channels=3,
                num_channel_initial=2,
                depth=None,
                extract_levels=None,
                out_kernel_initializer="he_normal",
                out_activation="softmax",
                pooling=False,
                concat_skip=False,
                encode_kernel_sizes=[7, 3, 3],
                decode_kernel_sizes=3,
                strides=2,
                padding="same",
                name="Test",
            )
        assert "GlobalNet requires `depth` or `extract_levels`" in str(err_info.value)

    def test_get_config(self):
        config = dict(
            image_size=(4, 5, 6),
            out_channels=3,
            num_channel_initial=2,
            depth=2,
            extract_levels=(2,),
            out_kernel_initializer="he_normal",
            out_activation="softmax",
            pooling=False,
            concat_skip=False,
            encode_kernel_sizes=[7, 3, 3],
            decode_kernel_sizes=3,
            encode_num_channels=[2, 4, 8],
            decode_num_channels=[2, 4, 8],
            strides=2,
            padding="same",
            name="Test",
        )
        network = GlobalNet(**config)
        got = network.get_config()
        assert got == config
