# coding=utf-8

"""
Tests for deepreg/model/network/ddf_dvf.py
"""
import pytest
import tensorflow as tf

from deepreg.model.network.ddf_dvf import build_ddf_dvf_model, ddf_dvf_forward
from deepreg.model.network.util import build_backbone


def test_ddf_dvf_forward():
    """
    Testing that ddf_dvf_forward function returns the tensors with correct shapes
    """

    moving_image_size = (1, 3, 5)
    fixed_image_size = (2, 4, 6)
    batch_size = 1

    local_net = build_backbone(
        image_size=fixed_image_size,
        out_channels=3,
        model_config={
            "backbone": "local",
            "local": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
        },
        method_name="ddf",
    )

    # Check DDF mode network output shapes - Pass
    dvf, ddf, pred_fixed_image, pred_fixed_label, grid_fixed = ddf_dvf_forward(
        backbone=local_net,
        moving_image=tf.ones((batch_size,) + moving_image_size),
        fixed_image=tf.ones((batch_size,) + fixed_image_size),
        moving_label=tf.ones((batch_size,) + moving_image_size),
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        output_dvf=False,
    )
    assert dvf is None
    assert ddf.shape == (batch_size,) + fixed_image_size + (3,)
    assert pred_fixed_image.shape == (batch_size,) + fixed_image_size
    assert pred_fixed_label.shape == (batch_size,) + fixed_image_size
    assert grid_fixed.shape == fixed_image_size + (3,)

    # Check DVF mode network output shapes - Pass
    dvf, ddf, pred_fixed_image, pred_fixed_label, grid_fixed = ddf_dvf_forward(
        backbone=local_net,
        moving_image=tf.ones((batch_size,) + moving_image_size),
        fixed_image=tf.ones((batch_size,) + fixed_image_size),
        moving_label=tf.ones((batch_size,) + moving_image_size),
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        output_dvf=True,
    )
    assert dvf.shape == (batch_size,) + fixed_image_size + (3,)
    assert ddf.shape == (batch_size,) + fixed_image_size + (3,)
    assert pred_fixed_image.shape == (batch_size,) + fixed_image_size
    assert pred_fixed_label.shape == (batch_size,) + fixed_image_size
    assert grid_fixed.shape == fixed_image_size + (3,)


def test_build_ddf_dvf_model():
    """
    Testing that build_ddf_dvf_model function returns the tensors with correct shapes
    """
    moving_image_size = (1, 3, 5)
    fixed_image_size = (2, 4, 6)
    batch_size = 1
    model_config = {
        "method": "ddf",
        "backbone": "local",
        "local": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
    }
    loss_config = {
        "dissimilarity": {
            "image": {"name": "lncc", "weight": 0.1},
            "label": {
                "name": "multi_scale",
                "weight": 1,
                "multi_scale": {
                    "loss_type": "dice",
                    "loss_scales": [0, 1, 2, 4, 8, 16, 32],
                },
            },
        },
        "regularization": {"weight": 0.0, "energy_type": "bending"},
    }

    # Create DDF model
    model_ddf = build_ddf_dvf_model(
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        index_size=1,
        labeled=True,
        batch_size=batch_size,
        model_config=model_config,
        loss_config=loss_config,
    )

    # Create DVF model
    model_config["method"] = "dvf"
    model_dvf = build_ddf_dvf_model(
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        index_size=1,
        labeled=True,
        batch_size=batch_size,
        model_config=model_config,
        loss_config=loss_config,
    )
    inputs = {
        "moving_image": tf.ones((batch_size,) + moving_image_size),
        "fixed_image": tf.ones((batch_size,) + fixed_image_size),
        "indices": 1,
        "moving_label": tf.ones((batch_size,) + moving_image_size),
        "fixed_label": tf.ones((batch_size,) + fixed_image_size),
    }
    outputs_ddf = model_ddf(inputs)
    outputs_dvf = model_dvf(inputs)

    expected_outputs_keys = ["dvf", "ddf", "pred_fixed_label"]
    assert all(keys in expected_outputs_keys for keys in outputs_ddf)
    assert outputs_ddf["pred_fixed_label"].shape == (batch_size,) + fixed_image_size
    assert outputs_ddf["ddf"].shape == (batch_size,) + fixed_image_size + (3,)
    with pytest.raises(KeyError):
        outputs_ddf["dvf"]

    assert all(keys in expected_outputs_keys for keys in outputs_dvf)
    assert outputs_dvf["pred_fixed_label"].shape == (batch_size,) + fixed_image_size
    assert outputs_dvf["dvf"].shape == (batch_size,) + fixed_image_size + (3,)
    assert outputs_dvf["ddf"].shape == (batch_size,) + fixed_image_size + (3,)
