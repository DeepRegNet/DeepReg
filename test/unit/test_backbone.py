# coding=utf-8

"""
Tests for deepreg/model/backbone
"""
from test.unit.util import is_equal_tf

import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.backbone as backbone
import deepreg.model.backbone.global_net as g
import deepreg.model.backbone.local_net as loc
import deepreg.model.backbone.u_net as u
import deepreg.model.layer as layer


def test_backbone_interface():
    """Test the get_config of the interface"""
    config = dict(
        image_size=(5, 5, 5),
        out_channels=3,
        num_channel_initial=4,
        out_kernel_initializer="zeros",
        out_activation="relu",
        name="test",
    )
    model = backbone.Backbone(**config)
    got = model.get_config()
    assert got == config


def test_init_global_net():
    """
    Testing init of GlobalNet is built as expected.
    """
    # initialising GlobalNet instance
    global_test = g.GlobalNet(
        image_size=[1, 2, 3],
        out_channels=3,
        num_channel_initial=3,
        extract_levels=[1, 2, 3],
        out_kernel_initializer="softmax",
        out_activation="softmax",
    )

    # asserting initialised var for extract_levels is the same - Pass
    assert global_test._extract_levels == [1, 2, 3]
    # asserting initialised var for extract_max_level is the same - Pass
    assert global_test._extract_max_level == 3

    # self reference grid
    # assert global_test.reference_grid correct shape, Pass
    assert global_test.reference_grid.shape == [1, 2, 3, 3]
    # assert correct reference grid returned, Pass
    expected_ref_grid = tf.convert_to_tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]],
                [[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 2.0]],
            ]
        ],
        dtype=tf.float32,
    )
    assert is_equal_tf(global_test.reference_grid, expected_ref_grid)

    # assert correct initial transform is returned
    expected_transform_initial = tf.convert_to_tensor(
        [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        dtype=tf.float32,
    )
    global_transform_initial = tf.Variable(global_test.transform_initial(shape=[12]))
    assert is_equal_tf(global_transform_initial, expected_transform_initial)

    # assert conv3dBlock type is correct, Pass
    assert isinstance(global_test._conv3d_block, layer.Conv3dBlock)


def test_call_global_net():
    """
    Asserting that output shape of globalnet Call method
    is correct.
    """
    out = 3
    im_size = (1, 2, 3)
    batch_size = 5
    # initialising GlobalNet instance
    global_test = g.GlobalNet(
        image_size=im_size,
        out_channels=out,
        num_channel_initial=3,
        extract_levels=[1, 2, 3],
        out_kernel_initializer="softmax",
        out_activation="softmax",
    )
    # pass an input of all zeros
    inputs = tf.constant(
        np.zeros(
            (batch_size, im_size[0], im_size[1], im_size[2], out), dtype=np.float32
        )
    )
    # get outputs by calling
    ddf, theta = global_test.call(inputs)
    assert ddf.shape == (batch_size, *im_size, 3)
    assert theta.shape == (batch_size, 4, 3)


class TestLocalNet:
    """
    Test the backbone.local_net.LocalNet class
    """

    @pytest.mark.parametrize(
        "image_size,extract_levels,control_points",
        [((1, 2, 3), [1, 2, 3], None), ((8, 8, 8), [1, 2, 3], (2, 2, 2))],
    )
    def test_init(self, image_size, extract_levels, control_points):
        network = loc.LocalNet(
            image_size=image_size,
            out_channels=3,
            num_channel_initial=3,
            extract_levels=extract_levels,
            out_kernel_initializer="he_normal",
            out_activation="softmax",
            control_points=control_points,
        )

        # asserting initialised var for extract_levels is the same - Pass
        assert network._extract_levels == extract_levels
        # asserting initialised var for extract_max_level is the same - Pass
        assert network._extract_max_level == max(extract_levels)
        # asserting initialised var for extract_min_level is the same - Pass
        assert network._extract_min_level == min(extract_levels)

        # assert number of downsample blocks is correct (== max level), Pass
        assert len(network._downsample_convs) == max(extract_levels)

        # assert upsample blocks type is correct, Pass
        assert all(
            isinstance(item, layer.LocalNetUpSampleResnetBlock)
            for item in network._upsample_blocks
        )
        # assert number of upsample blocks is correct (== max level - min level), Pass
        assert len(network._upsample_blocks) == max(extract_levels) - min(
            extract_levels
        )

        # assert number of upsample blocks is correct (== extract_levels), Pass
        assert len(network._extract_layers) == len(extract_levels)

        if control_points is None:
            assert network.resize is False
        else:
            assert isinstance(network.resize, layer.ResizeCPTransform)
            assert isinstance(network.interpolate, layer.BSplines3DTransform)

    @pytest.mark.parametrize(
        "image_size,extract_levels,control_points",
        [((1, 2, 3), [1, 2, 3], None), ((8, 8, 8), [1, 2, 3], (2, 2, 2))],
    )
    def test_call(self, image_size, extract_levels, control_points):
        # initialising LocalNet instance
        network = loc.LocalNet(
            image_size=image_size,
            out_channels=3,
            num_channel_initial=3,
            extract_levels=extract_levels,
            out_kernel_initializer="he_normal",
            out_activation="softmax",
            control_points=control_points,
        )

        # pass an input of all zeros
        inputs = tf.constant(
            np.zeros(
                (5, image_size[0], image_size[1], image_size[2], 3), dtype=np.float32
            )
        )
        # get outputs by calling
        output = network.call(inputs)
        # expected shape is (5, 1, 2, 3, 3)
        assert all(x == y for x, y in zip(inputs.shape, output.shape))


class TestUNet:
    """
    Test the backbone.u_net.UNet class
    """

    @pytest.mark.parametrize(
        "image_size,depth,control_points",
        [((1, 2, 3), 5, None), ((8, 8, 8), 3, (2, 2, 2))],
    )
    def test_init(self, image_size, depth, control_points):
        network = u.UNet(
            image_size=image_size,
            out_channels=3,
            num_channel_initial=2,
            depth=depth,
            out_kernel_initializer="he_normal",
            out_activation="softmax",
            control_points=control_points,
        )

        # asserting num channels initial is the same, Pass
        assert network._num_channel_initial == 2

        # asserting depth is the same, Pass
        assert network._depth == depth

        # assert bottom_conv3d type is correct, Pass
        assert isinstance(network._bottom_conv3d, layer.Conv3dBlock)

        # assert upsample blocks type is correct, Pass
        assert all(
            isinstance(item, layer.UpSampleResnetBlock)
            for item in network._upsample_blocks
        )
        # assert number of upsample blocks is correct (== depth), Pass
        assert len(network._upsample_blocks) == depth

        if control_points is None:
            assert network.resize is False
        else:
            assert isinstance(network.resize, layer.ResizeCPTransform)
            assert isinstance(network.interpolate, layer.BSplines3DTransform)

    @pytest.mark.parametrize(
        "image_size,depth,control_points",
        [((1, 2, 3), 5, None), ((8, 8, 8), 3, (2, 2, 2))],
    )
    def test_call_unet(self, image_size, depth, control_points):
        out = 3
        # initialising UNet instance
        network = u.UNet(
            image_size=image_size,
            out_channels=3,
            num_channel_initial=2,
            depth=depth,
            out_kernel_initializer="he_normal",
            out_activation="softmax",
            control_points=control_points,
        )
        # pass an input of all zeros
        inputs = tf.constant(
            np.zeros(
                (5, image_size[0], image_size[1], image_size[2], out), dtype=np.float32
            )
        )
        # get outputs by calling
        output = network.call(inputs)
        # expected shape is (5, 1, 2, 3)
        assert all(x == y for x, y in zip(inputs.shape, output.shape))
