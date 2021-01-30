# coding=utf-8

"""
Tests for deepreg/model/layer
"""
from typing import Optional

import numpy as np
import pytest
import tensorflow as tf

import deepreg.model.layer as layer


@pytest.mark.parametrize(
    ("input_shape", "output_shape", "expected_shape"),
    [
        ((6, 7, 8), (12, 14, 16), (12, 14, 16)),
        ((6, 7, 8), (11, 13, 15), (11, 13, 15)),
        ((6, 7, 8), None, (12, 14, 16)),
    ],
)
def test_deconv3d(input_shape, output_shape: Optional[tuple], expected_shape: tuple):
    """
    Test output shapes and configs.

    :param input_shape: input shape for layer definition
    :param output_shape: output shape for layer definition
    :param expected_shape: expected output shape
    """
    batch_size = 5

    deconv3d = layer.Deconv3d(filters=3, strides=2, output_shape=output_shape)

    inputs = tf.ones(shape=(batch_size,) + input_shape + (1,))
    output = deconv3d(inputs)

    assert output.shape == (batch_size,) + expected_shape + (3,)

    config = deconv3d.get_config()
    assert config == dict(
        filters=3,
        output_shape=output_shape,
        kernel_size=3,
        strides=2,
        padding="same",
        name="deconv3d",
        trainable=True,
        dtype="float32",
    )


@pytest.mark.parametrize("layer_name", ["conv3d", "deconv3d"])
@pytest.mark.parametrize("norm_name", ["batch", "layer"])
@pytest.mark.parametrize("activation", ["relu", "elu"])
def test_norm_block(layer_name: str, norm_name: str, activation: str):
    """
    Test output shapes and configs.

    :param layer_name: layer_name for layer definition
    :param norm_name: norm_name for layer definition
    :param activation: activation for layer definition
    """
    input_size = (2, 3, 4, 5, 6)  # (batch, *shape, ch)
    conv_block = layer.NormBlock(
        layer_name=layer_name,
        norm_name=norm_name,
        activation=activation,
        filters=3,
        kernel_size=1,
        padding="same",
    )
    inputs = tf.ones(shape=input_size)
    outputs = conv_block(inputs)
    assert outputs.shape == input_size[:-1] + (3,)

    config = conv_block.get_config()
    assert config == dict(
        layer_name=layer_name,
        norm_name=norm_name,
        activation=activation,
        filters=3,
        kernel_size=1,
        padding="same",
        name="norm_block",
        trainable=True,
        dtype="float32",
    )


def test_deconv3d_block():
    """
    Test the layer.Deconv3dBlock class and its default attributes.
    """

    deconv3d_block = layer.Deconv3dBlock(8)

    assert isinstance(deconv3d_block._deconv3d, layer.Deconv3d)
    assert deconv3d_block._deconv3d._deconv3d is None

    deconv3d_block._deconv3d.build((8, 8))


def test_residual3d_block():
    """
    Test the layer.Residual3dBlock class and its default attributes.
    """
    res3d_block = layer.Residual3dBlock(8)

    assert isinstance(res3d_block._conv3d_block, layer.Conv3dBlock)
    assert res3d_block._conv3d_block._conv3d.kernel_size == (3, 3, 3)
    assert res3d_block._conv3d_block._conv3d.strides == (1, 1, 1)

    assert res3d_block._conv3d.use_bias is False
    assert res3d_block._conv3d.kernel_size == (3, 3, 3)
    assert res3d_block._conv3d.strides == (1, 1, 1)


def test_downsample_resnet_block():
    """
    Test the layer.DownSampleResnetBlock class and its default attributes.
    """
    model = layer.DownSampleResnetBlock(8)

    assert model._pooling is True

    assert isinstance(model._residual_block, layer.Residual3dBlock)
    assert model._conv3d_block3 is None

    model = layer.DownSampleResnetBlock(8, pooling=False)
    assert model._max_pool3d is None
    assert isinstance(model._conv3d_block3, layer.Conv3dBlock)


def test_upsample_resnet_block():
    """
    Test the layer.UpSampleResnetBlock class and its default attributes.
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


def test_init_conv3d_with_resize():
    """
    Test the layer.Conv3dWithResize class's default attributes and call function.
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


def test_init_dvf():
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


def test_additive_upsampling():
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


def test_local_net_residual3d_block():
    """
    Test the layer.LocalNetResidual3dBlock class's,
    default attributes and call() function.
    """

    # Test __init__()
    conv3d_block = layer.LocalNetResidual3dBlock(8)

    assert conv3d_block._conv3d.kernel_size == (3, 3, 3)
    assert conv3d_block._conv3d.strides == (1, 1, 1)
    assert conv3d_block._conv3d.padding == "same"
    assert conv3d_block._conv3d.use_bias is False


def test_local_net_upsample_resnet_block():
    """
    Test the layer.LocalNetUpSampleResnetBlock class,
    its default attributes and its call() function.
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


class TestResizeCPTransform:
    @pytest.mark.parametrize(
        "parameter,cp_spacing", [((8, 8, 8), 8), ((8, 24, 16), (8, 24, 16))]
    )
    def test_attributes(self, parameter, cp_spacing):
        model = layer.ResizeCPTransform(cp_spacing)

        if isinstance(cp_spacing, int):
            cp_spacing = [cp_spacing] * 3
        assert list(model.cp_spacing) == list(parameter)
        assert model.kernel_sigma == [0.44 * cp for cp in cp_spacing]

    @pytest.mark.parametrize(
        "input_size,output_size,cp_spacing",
        [
            ((1, 8, 8, 8, 3), (12, 8, 12), (8, 16, 8)),
            ((1, 8, 8, 8, 3), (12, 12, 12), 8),
        ],
    )
    def test_build(self, input_size, output_size, cp_spacing):
        model = layer.ResizeCPTransform(cp_spacing)
        model.build(input_size)

        assert [a == b for a, b, in zip(model._output_shape, output_size)]

    @pytest.mark.parametrize(
        "input_size,output_size,cp_spacing",
        [
            ((1, 68, 68, 68, 3), (1, 12, 8, 12, 3), (8, 16, 8)),
            ((1, 68, 68, 68, 3), (1, 12, 12, 12, 3), 8),
        ],
    )
    def test_call(self, input_size, output_size, cp_spacing):
        model = layer.ResizeCPTransform(cp_spacing)
        model.build(input_size)

        input = tf.random.normal(shape=input_size, dtype=tf.float32)
        output = model(input)

        assert output.shape == output_size


class TestBSplines3DTransform:
    """
    Test the layer.BSplines3DTransform class,
    its default attributes and its call() function.
    """

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), 8), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def test_init(self, input_size, cp):
        model = layer.BSplines3DTransform(cp, input_size[1:-1])

        if isinstance(cp, int):
            cp = (cp, cp, cp)

        assert model.cp_spacing == cp

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), (8, 8, 8)), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def generate_filter_coefficients(self, cp_spacing):

        b = {
            0: lambda u: np.float64((1 - u) ** 3 / 6),
            1: lambda u: np.float64((3 * (u ** 3) - 6 * (u ** 2) + 4) / 6),
            2: lambda u: np.float64((-3 * (u ** 3) + 3 * (u ** 2) + 3 * u + 1) / 6),
            3: lambda u: np.float64(u ** 3 / 6),
        }

        filters = np.zeros(
            (
                4 * cp_spacing[0],
                4 * cp_spacing[1],
                4 * cp_spacing[2],
                3,
                3,
            ),
            dtype=np.float32,
        )

        for u in range(cp_spacing[0]):
            for v in range(cp_spacing[1]):
                for w in range(cp_spacing[2]):
                    for x in range(4):
                        for y in range(4):
                            for z in range(4):
                                for it_dim in range(3):
                                    u_norm = 1 - (u + 0.5) / cp_spacing[0]
                                    v_norm = 1 - (v + 0.5) / cp_spacing[1]
                                    w_norm = 1 - (w + 0.5) / cp_spacing[2]
                                    filters[
                                        x * cp_spacing[0] + u,
                                        y * cp_spacing[1] + v,
                                        z * cp_spacing[2] + w,
                                        it_dim,
                                        it_dim,
                                    ] = (
                                        b[x](u_norm) * b[y](v_norm) * b[z](w_norm)
                                    )
        return filters

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), (8, 8, 8)), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def test_build(self, input_size, cp):
        model = layer.BSplines3DTransform(cp, input_size[1:-1])

        model.build(input_size)
        assert model.filter.shape == (
            4 * cp[0],
            4 * cp[1],
            4 * cp[2],
            3,
            3,
        )

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), (8, 8, 8)), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def test_coefficients(self, input_size, cp):

        filters = self.generate_filter_coefficients(cp_spacing=cp)

        model = layer.BSplines3DTransform(cp, input_size[1:-1])
        model.build(input_size)

        assert np.allclose(filters, model.filter.numpy(), atol=1e-8)

    @pytest.mark.parametrize(
        "input_size,cp",
        [((1, 8, 8, 8, 3), (8, 8, 8)), ((1, 8, 8, 8, 3), (8, 16, 12))],
    )
    def test_interpolation(self, input_size, cp):
        model = layer.BSplines3DTransform(cp, input_size[1:-1])
        model.build(input_size)

        vol_shape = input_size[1:-1]
        num_cp = (
            [input_size[0]]
            + [int(np.ceil(isize / cpsize) + 3) for isize, cpsize in zip(vol_shape, cp)]
            + [input_size[-1]]
        )

        field = tf.random.normal(shape=num_cp, dtype=tf.float32)

        ddf = model.call(field)
        assert ddf.shape == input_size
