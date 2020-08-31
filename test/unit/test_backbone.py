# coding=utf-8

"""
Tests for deepreg/model/backbone
"""
from test.unit.util import is_equal_tf

import numpy as np
import tensorflow as tf

import deepreg.model.backbone.global_net as g
import deepreg.model.backbone.local_net as loc
import deepreg.model.backbone.u_net as u
import deepreg.model.layer as layer


def test_init_GlobalNet():
    """
    Testing init of GlobalNet is built as expected.
    """
    # Initialising GlobalNet instance
    global_test = g.GlobalNet(
        image_size=[1, 2, 3],
        out_channels=3,
        num_channel_initial=3,
        extract_levels=[1, 2, 3],
        out_kernel_initializer="softmax",
        out_activation="softmax",
    )

    # Asserting initialised var for extract_levels is the same - Pass
    assert global_test._extract_levels == [1, 2, 3]
    # Asserting initialised var for extract_max_level is the same - Pass
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

    # Testing constant initializer
    # We initialize the expected tensor and initialise another from the
    # class variable using tf.Variable
    test_tensor_return = tf.convert_to_tensor(
        [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        dtype=tf.float32,
    )
    global_return = tf.Variable(
        global_test.transform_initial(shape=[6, 2], dtype=tf.float32)
    )

    # Asserting they are equal - Pass
    assert is_equal_tf(
        test_tensor_return, tf.convert_to_tensor(global_return, dtype=tf.float32)
    )

    # Assert downsample blocks type is correct, Pass
    assert all(
        isinstance(item, layer.DownSampleResnetBlock)
        for item in global_test._downsample_blocks
    )
    # Assert number of downsample blocks is correct (== max level), Pass
    assert len(global_test._downsample_blocks) == 3

    #  Assert conv3dBlock type is correct, Pass
    assert isinstance(global_test._conv3d_block, layer.Conv3dBlock)

    #  Asserting type is dense_layer, Pass
    assert isinstance(global_test._dense_layer, layer.Dense)


def test_call_GlobalNet():
    """
    Asserting that output shape of globalnet Call method
    is correct.
    """
    out = 3
    im_size = [1, 2, 3]
    # Initialising GlobalNet instance
    global_test = g.GlobalNet(
        image_size=im_size,
        out_channels=out,
        num_channel_initial=3,
        extract_levels=[1, 2, 3],
        out_kernel_initializer="softmax",
        out_activation="softmax",
    )
    # Pass an input of all zeros
    inputs = tf.constant(
        np.zeros((5, im_size[0], im_size[1], im_size[2], out), dtype=np.float32)
    )
    # Get outputs by calling
    output = global_test.call(inputs)
    # Expected shape is (5, 1, 2, 3, 3)
    assert all(x == y for x, y in zip(inputs.shape, output.shape))


# Testing LocalNet
def test_init_LocalNet():
    """
    Testing init of LocalNet as expected
    """
    local_test = loc.LocalNet(
        image_size=[1, 2, 3],
        out_channels=3,
        num_channel_initial=3,
        extract_levels=[1, 2, 3],
        out_kernel_initializer="he_normal",
        out_activation="softmax",
    )

    # Asserting initialised var for extract_levels is the same - Pass
    assert local_test._extract_levels == [1, 2, 3]
    # Asserting initialised var for extract_max_level is the same - Pass
    assert local_test._extract_max_level == 3
    # Asserting initialised var for extract_min_level is the same - Pass
    assert local_test._extract_min_level == 1

    # Assert downsample blocks type is correct, Pass
    assert all(
        isinstance(item, layer.DownSampleResnetBlock)
        for item in local_test._downsample_blocks
    )
    # Assert number of downsample blocks is correct (== max level), Pass
    assert len(local_test._downsample_blocks) == 3

    # Assert upsample blocks type is correct, Pass
    assert all(
        isinstance(item, layer.LocalNetUpSampleResnetBlock)
        for item in local_test._upsample_blocks
    )
    # Assert number of upsample blocks is correct (== max level - min level), Pass
    assert len(local_test._upsample_blocks) == 3 - 1

    # Assert upsample blocks type is correct, Pass
    assert all(
        isinstance(item, layer.Conv3dWithResize) for item in local_test._extract_layers
    )
    # Assert number of upsample blocks is correct (== extract_levels), Pass
    assert len(local_test._extract_layers) == 3


def test_call_LocalNet():
    """
    Asserting that output shape of LocalNet call method
    is correct.
    """
    out = 3
    im_size = [1, 2, 3]
    # Initialising LocalNet instance
    global_test = loc.LocalNet(
        image_size=im_size,
        out_channels=out,
        num_channel_initial=3,
        extract_levels=[1, 2, 3],
        out_kernel_initializer="glorot_uniform",
        out_activation="sigmoid",
    )
    # Pass an input of all zeros
    inputs = tf.constant(
        np.zeros((5, im_size[0], im_size[1], im_size[2], out), dtype=np.float32)
    )
    # Get outputs by calling
    output = global_test.call(inputs)
    # Expected shape is (5, 1, 2, 3, 3)
    assert all(x == y for x, y in zip(inputs.shape, output.shape))


# Testing UNet
def test_init_UNet():
    """
    Testing init of UNet as expected
    """
    local_test = u.UNet(
        image_size=[1, 2, 3],
        out_channels=3,
        num_channel_initial=3,
        depth=5,
        out_kernel_initializer="he_normal",
        out_activation="softmax",
    )

    # Asserting num channels initial is the same, Pass
    assert local_test._num_channel_initial == 3

    # Asserting depth is the same, Pass
    assert local_test._depth == 5

    # Assert downsample blocks type is correct, Pass
    assert all(
        isinstance(item, layer.DownSampleResnetBlock)
        for item in local_test._downsample_blocks
    )
    # Assert number of downsample blocks is correct (== depth), Pass
    assert len(local_test._downsample_blocks) == 5

    #  Assert bottom_conv3d type is correct, Pass
    assert isinstance(local_test._bottom_conv3d, layer.Conv3dBlock)

    # Assert bottom res3d type is correct, Pass
    assert isinstance(local_test._bottom_res3d, layer.Residual3dBlock)
    # Assert upsample blocks type is correct, Pass
    assert all(
        isinstance(item, layer.UpSampleResnetBlock)
        for item in local_test._upsample_blocks
    )
    # Assert number of upsample blocks is correct (== depth), Pass
    assert len(local_test._upsample_blocks) == 5

    # Assert output_conv3d is correct type, Pass
    assert isinstance(local_test._output_conv3d, layer.Conv3dWithResize)


def test_call_UNet():
    """
    Asserting that output shape of UNet call method
    is correct.
    """
    out = 3
    im_size = [1, 2, 3]
    # Initialising LocalNet instance
    global_test = u.UNet(
        image_size=im_size,
        out_channels=out,
        num_channel_initial=3,
        depth=6,
        out_kernel_initializer="glorot_uniform",
        out_activation="sigmoid",
    )
    # Pass an input of all zeros
    inputs = tf.constant(
        np.zeros((5, im_size[0], im_size[1], im_size[2], out), dtype=np.float32)
    )
    # Get outputs by calling
    output = global_test.call(inputs)
    # Expected shape is (5, 1, 2, 3)
    assert all(x == y for x, y in zip(inputs.shape, output.shape))
