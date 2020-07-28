# coding=utf-8

"""
Tests for deepreg/model/network/util
"""
import pytest
import tensorflow as tf

import deepreg.model.network.util as util
from deepreg.model.backbone import global_net, local_net, u_net


def test_wrong_inputs():
    """
    Function to test wrong input types passed to build backbone func
    """
    #  Wrong image_size type: int, vs tuple, Fail
    with pytest.raises(Exception):
        util.build_backbone(
            image_size=1, out_channels=1, model_config={}, method_name="ddf"
        )
    #  Wrong out_channels type: str, vs int, Fail
    with pytest.raises(Exception):
        util.build_backbone(
            image_size=(1, 2, 3), out_channels="", model_config={}, method_name="ddf"
        )
    #  Wrong out_channels type: list, vs dic, Fail
    with pytest.raises(Exception):
        util.build_backbone(
            image_size=(1, 2, 3), out_channels=1, model_config=[], method_name="ddf"
        )
    #  Wrong out_channels type: int, vs str, Fail
    with pytest.raises(Exception):
        util.build_backbone(
            image_size=(1, 2, 3), out_channels=1, model_config={}, method_name=1
        )


def test_value_raised_if_wrong_method():
    """
    Checking ValueError raised if string not
    in accepted methods name
    """
    #  expect ddf, dvf or conditional, Fail
    with pytest.raises(ValueError):
        util.build_backbone(
            image_size=(1, 2, 3), out_channels=1, model_config={}, method_name=""
        )


def test_value_raised_if_unknown_config():
    """
    Checking ValueError raised if string for
    backbone unknown
    """
    #  expect local, global or unet
    with pytest.raises(ValueError):
        util.build_backbone(
            image_size=(1, 2, 3),
            out_channels=1,
            model_config={"backbone": "random"},
            method_name="ddf",
        )


def test_global_return():
    """
    Testing that build_backbone func returns an object
    of type GlobalNet from backbone module when initialised
    with the associated GlobalNet config.
    """
    out = util.build_backbone(
        image_size=(1, 2, 3),
        out_channels=1,
        model_config={
            "backbone": "global",
            "global": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
        },
        method_name="ddf",
    )
    assert isinstance(
        out,
        type(global_net.GlobalNet([1, 2, 3], 4, 4, [1, 2, 3], "he_normal", "sigmoid")),
    )


def test_local_return():
    """
    Testing that build_backbone func returns an object
    of type LocalNet from backbone module when initialised
    with the associated LocalNet config.
    """
    out = util.build_backbone(
        image_size=(1, 2, 3),
        out_channels=1,
        model_config={
            "backbone": "local",
            "local": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
        },
        method_name="ddf",
    )
    assert isinstance(
        out,
        type(local_net.LocalNet([1, 2, 3], 4, 4, [1, 2, 3], "he_normal", "sigmoid")),
    )


def test_unet_return():
    """
    Testing that build_backbone func returns an object
    of type UNet form backbone module when initialised
    with the associated UNet config.
    """
    out = util.build_backbone(
        image_size=(1, 2, 3),
        out_channels=1,
        model_config={
            "backbone": "unet",
            "unet": {"num_channel_initial": 4, "depth": 4},
        },
        method_name="ddf",
    )
    assert isinstance(out, type(u_net.UNet([1, 2, 3], 4, 4, 4, "he_normal", "sigmoid")))


def test_wrong_inputs_build_inputs():
    """
    Function to test wrong input types passed to build backbone func
    """
    #  Wrong image_size type: int, vs tuple, Fail
    with pytest.raises(Exception):
        util.build_inputs(
            moving_image_size=1,
            fixed_image_size=(),
            index_size=1,
            batch_size=1,
            labeled=True,
        )
    #  Wrong fixed_images type: int, vs tuple, Fail
    with pytest.raises(Exception):
        util.build_inputs(
            moving_image_size=(),
            fixed_image_size=1,
            index_size=1,
            batch_size=1,
            labeled=True,
        )

    #  Wrong indx_sie type: list, vs int, Fail
    with pytest.raises(Exception):
        util.build_inputs(
            moving_image_size=(),
            fixed_image_size=(),
            index_size=[],
            batch_size=1,
            labeled=True,
        )
    #  Wrong batch_size type: list, vs int, Fail
    with pytest.raises(Exception):
        util.build_inputs(
            moving_image_size=(),
            fixed_image_size=(),
            index_size=1,
            batch_size=[],
            labeled=True,
        )


def test_return_types_build_inputs():
    """
    Test that returns 5 items of type tf.keras.inputs.
    """
    out = util.build_inputs(
        moving_image_size=(1, 2, 3),
        fixed_image_size=(1, 2, 3),
        index_size=1,
        batch_size=1,
        labeled=True,
    )
    #  Asserting all items tf.keras.inputs - Pass
    assert all(isinstance(item, type(tf.keras.Input(1))) for item in out)

    mov_im, fixed_im, mov_l, fixed_l, indices = util.build_inputs(
        moving_image_size=(1, 2, 3),
        fixed_image_size=(1, 2, 3),
        index_size=1,
        batch_size=1,
        labeled=False,
    )
    #  Asserting all items bar mov_l and fixed_l tf.keras.inputs - Pass
    assert all(
        isinstance(item, type(tf.keras.Input(1)))
        for item in [mov_im, fixed_im, indices]
    )
    assert all(isinstance(item, type(None)) for item in [mov_l, fixed_l])
