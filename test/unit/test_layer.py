# coding=utf-8

"""
Tests for deepreg/model/layer
"""
import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.layer as layer


def test_activation():
    """
    Test the layer.Activation class and its default attributes. No need to test the call() function since its
    a tensorflow class
    """
    activation = layer.Activation()
    assert isinstance(activation._act, type(tf.keras.activations.relu))


def test_norm():
    """
    Test the layer.Norm class, its default attributes and errors raised. No need to test the call() function since its
    a tensorflow class
    """

    norm = layer.Norm()
    assert isinstance(norm._norm, tf.keras.layers.BatchNormalization)

    with pytest.raises(ValueError):
        layer.Norm(name="none")


def test_maxpool3d():
    """
    Test the layer.MaxPool3d class and its default attributes. No need to test the call() function since its
    a tensorflow class
    """
    pooling = layer.MaxPool3d(2)

    assert isinstance(pooling._max_pool, tf.keras.layers.MaxPool3D)
    assert pooling._max_pool.pool_function == tf.nn.max_pool3d
    assert pooling._max_pool.strides == (2, 2, 2)
    assert pooling._max_pool.padding == "same"


def test_conv3d():
    """
    Test the layer.Conv3d class and its default attributes. No need to test the call() function since its
    a tensorflow class
    """

    conv3d = layer.Conv3d(8)

    assert isinstance(conv3d._conv3d, tf.keras.layers.Conv3D)
    assert conv3d._conv3d.kernel_size == (3, 3, 3)
    assert conv3d._conv3d.strides == (1, 1, 1)
    assert conv3d._conv3d.padding == "same"
    assert isinstance(conv3d._conv3d.activation, type(tf.keras.activations.linear))
    assert conv3d._conv3d.use_bias is True


def test_deconv3d():
    """
    Test the layer.Deconv3d class and its default attributes. No need to test the call() function since its
    a tensorflow class
    """
    batch_size = 5
    channels = 4
    input_size = (32, 32, 16)
    output_size = (64, 64, 32)
    output_padding = (1, 1, 1)

    input_tensor_shape = (batch_size,) + input_size + (channels,)

    deconv3d = layer.Deconv3d(8, output_size, strides=2)
    deconv3d.build(input_tensor_shape)

    assert tuple(deconv3d._output_padding) == output_padding
    assert isinstance(deconv3d._Conv3DTranspose, tf.keras.layers.Conv3DTranspose)
    assert tuple(deconv3d._kernel_size) == (3, 3, 3)
    assert tuple(deconv3d._strides) == (2, 2, 2)
    assert deconv3d._padding == "same"
    assert deconv3d._Conv3DTranspose.use_bias is True


def test_conv3dBlock():
    """
    Test the layer.Conv3dBlock class and its default attributes. No need to test the call() function since its
    a tensorflow class
    """

    conv3dBlock = layer.Conv3dBlock(8)

    assert isinstance(conv3dBlock._conv3d, layer.Conv3d)

    assert conv3dBlock._conv3d._conv3d.kernel_size == (3, 3, 3)
    assert conv3dBlock._conv3d._conv3d.strides == (1, 1, 1)
    assert conv3dBlock._conv3d._conv3d.padding == "same"
    assert conv3dBlock._conv3d._conv3d.use_bias is False

    assert isinstance(conv3dBlock._act._act, type(tf.keras.activations.relu))
    assert isinstance(conv3dBlock._norm._norm, tf.keras.layers.BatchNormalization)


def test_deconv3dBlock():
    """
    Test the layer.Deconv3dBlock class and its default attributes. No need to test the call() function since a
    concatenation of tensorflow classes
    """

    deconv3dBlock = layer.Deconv3dBlock(8)

    assert isinstance(deconv3dBlock._deconv3d, layer.Deconv3d)
    assert deconv3dBlock._deconv3d._Conv3DTranspose is None

    deconv3dBlock._deconv3d.build((8, 8))

    assert isinstance(
        deconv3dBlock._deconv3d._Conv3DTranspose, tf.keras.layers.Conv3DTranspose
    )
    assert tuple(deconv3dBlock._deconv3d._kernel_size) == (3, 3, 3)
    assert tuple(deconv3dBlock._deconv3d._strides) == (1, 1, 1)
    assert deconv3dBlock._deconv3d._padding == "same"
    assert deconv3dBlock._deconv3d._Conv3DTranspose.use_bias is False

    assert isinstance(deconv3dBlock._act._act, type(tf.keras.activations.relu))
    assert isinstance(deconv3dBlock._norm._norm, tf.keras.layers.BatchNormalization)


def test_residual3dBlock():
    """
    Test the layer.Residual3dBlock class and its default attributes. No need to test the call() function since a
    concatenation of tensorflow classes
    """
    res3dBlock = layer.Residual3dBlock(8)

    assert isinstance(res3dBlock._conv3d_block, layer.Conv3dBlock)
    assert res3dBlock._conv3d_block._conv3d._conv3d.kernel_size == (3, 3, 3)
    assert res3dBlock._conv3d_block._conv3d._conv3d.strides == (1, 1, 1)

    assert isinstance(res3dBlock._conv3d, layer.Conv3d)
    assert res3dBlock._conv3d._conv3d.use_bias is False
    assert res3dBlock._conv3d._conv3d.kernel_size == (3, 3, 3)
    assert res3dBlock._conv3d._conv3d.strides == (1, 1, 1)

    assert isinstance(res3dBlock._act._act, type(tf.keras.activations.relu))
    assert isinstance(res3dBlock._norm._norm, tf.keras.layers.BatchNormalization)


def test_downsampleResnetBlock():
    """
    Test the layer.DownSampleResnetBlock class and its default attributes. No need to test the call() function since a
    concatenation of tensorflow classes
    """
    model = layer.DownSampleResnetBlock(8)

    assert model._pooling is True

    assert isinstance(model._conv3d_block, layer.Conv3dBlock)
    assert isinstance(model._residual_block, layer.Residual3dBlock)
    assert isinstance(model._max_pool3d, layer.MaxPool3d)
    assert model._conv3d_block3 is None

    model = layer.DownSampleResnetBlock(8, pooling=False)
    assert model._max_pool3d is None
    assert isinstance(model._conv3d_block3, layer.Conv3dBlock)


def test_upsampleResnetBlock():
    """
    Test the layer.UpSampleResnetBlock class and its default attributes. No need to test the call() function since a
    concatenation of tensorflow classes
    """
    batch_size = 5
    channels = 4
    input_size = (32, 32, 16)
    output_size = (64, 64, 32)

    input_tensor_size = (batch_size,) + input_size + (channels,)
    skip_tensor_size = (batch_size,) + output_size + (channels // 2,)

    model = layer.UpSampleResnetBlock(8)
    model.build([input_tensor_size, skip_tensor_size])

    assert model._filters == 8
    assert model._concat is False
    assert isinstance(model._conv3d_block, layer.Conv3dBlock)
    assert isinstance(model._residual_block, layer.Residual3dBlock)
    assert isinstance(model._deconv3d_block, layer.Deconv3dBlock)


def test_init_conv3dWithResize():
    """
    Test the layer.Conv3dWithResize class, its default attributes and it's call function.
    """
    batch_size = 5
    channels = 4
    input_size = (32, 32, 16)
    output_size = (62, 62, 24)
    filters = 8

    input_tensor_size = (batch_size,) + input_size + (channels,)
    output_tensor_size = (batch_size,) + output_size + (filters,)

    model = layer.Conv3dWithResize(output_size, filters)

    assert model._output_shape == output_size
    assert isinstance(model._conv3d, layer.Conv3d)

    # Pass an input of all zeros
    inputs = np.zeros(input_tensor_size)
    # Get outputs by calling
    output = model.call(inputs)
    # Expected shape is (5, 1, 2, 3, 3)
    assert all(x == y for x, y in zip(output_tensor_size, output.shape))


def test_warping():
    """
    Test the layer.Warping class, its default attributes and its call() method.
    """
    batch_size = 5
    fixed_image_size = (32, 32, 16)
    moving_image_size = (24, 24, 16)
    ndims = len(moving_image_size)

    grid_size = (1,) + fixed_image_size + (3,)
    model = layer.Warping(fixed_image_size)

    assert all(x == y for x, y in zip(grid_size, model.grid_ref.shape))

    # Pass an input of all zeros
    inputs = [
        np.ones((batch_size, *fixed_image_size, ndims), dtype="float32"),
        np.ones((batch_size, *moving_image_size), dtype="float32"),
    ]
    # Get outputs by calling
    output = model.call(inputs)
    # Expected shape is (5, 1, 2, 3, 3)
    assert all(x == y for x, y in zip((batch_size,) + fixed_image_size, output.shape))


def test_initDVF():
    """
    Test the layer.IntDVF class, its default attributes and its call() method.
    """

    batch_size = 5
    fixed_image_size = (32, 32, 16)
    ndims = len(fixed_image_size)

    model = layer.IntDVF(fixed_image_size)

    assert isinstance(model._warping, layer.Warping)
    assert model._num_steps == 7

    inputs = np.ones((batch_size, *fixed_image_size, ndims))
    output = model.call(inputs)
    assert all(
        x == y
        for x, y in zip((batch_size,) + fixed_image_size + (ndims,), output.shape)
    )


def test_dense():
    """
    Test the layer.Dense class and its default attributes. No need to test the call() function since a
    concatenation of tensorflow classes
    """
    model = layer.Dense(8)

    assert isinstance(model._flatten, tf.keras.layers.Flatten)
    assert isinstance(model._dense, tf.keras.layers.Dense)


def test_additiveUpSampling():
    """
    Test the layer.AdditiveUpSampling class and its default attributes.
    """
    channels = 8
    batch_size = 5
    output_size = (32, 32, 16)
    input_size = (24, 24, 16)

    # Test __init__()
    model = layer.AdditiveUpSampling(output_size)
    assert model._stride == 2
    assert model._output_shape == output_size

    # Test call()
    inputs = np.ones(
        (batch_size, input_size[0], input_size[1], input_size[2], channels)
    )
    output = model(inputs)
    assert all(
        x == y
        for x, y in zip((batch_size,) + output_size + (channels / 2,), output.shape)
    )

    # Test the exceptions
    model = layer.AdditiveUpSampling(output_size, stride=3)
    with pytest.raises(ValueError):
        model(inputs)


def test_localNetResidual3dBlock():
    """
    Test the layer.LocalNetResidual3dBlock class, its default attributes and its call() function. No need to test
    the call() function since its a concatenation of tensorflow classes.
    """

    # Test __init__()
    conv3dBlock = layer.LocalNetResidual3dBlock(8)

    assert isinstance(conv3dBlock._conv3d, layer.Conv3d)

    assert conv3dBlock._conv3d._conv3d.kernel_size == (3, 3, 3)
    assert conv3dBlock._conv3d._conv3d.strides == (1, 1, 1)
    assert conv3dBlock._conv3d._conv3d.padding == "same"
    assert conv3dBlock._conv3d._conv3d.use_bias is False

    assert isinstance(conv3dBlock._act._act, type(tf.keras.activations.relu))
    assert isinstance(conv3dBlock._norm._norm, tf.keras.layers.BatchNormalization)


def test_localNetUpSampleResnetBlock():
    """
    Test the layer.LocalNetUpSampleResnetBlock class, its default attributes and its call() function.
    """
    batch_size = 5
    channels = 4
    input_size = (32, 32, 16)
    output_size = (64, 64, 32)

    nonskip_tensor_size = (batch_size,) + input_size + (channels,)
    skip_tensor_size = (batch_size,) + output_size + (channels,)

    # Test __init__() and build()
    model = layer.LocalNetUpSampleResnetBlock(8)
    model.build([nonskip_tensor_size, skip_tensor_size])

    assert model._filters == 8
    assert model._use_additive_upsampling is True

    assert isinstance(model._deconv3d_block, layer.Deconv3dBlock)
    assert isinstance(model._additive_upsampling, layer.AdditiveUpSampling)
    assert isinstance(model._conv3d_block, layer.Conv3dBlock)
    assert isinstance(model._residual_block, layer.LocalNetResidual3dBlock)
