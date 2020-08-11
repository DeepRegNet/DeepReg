# coding=utf-8

"""
Tests for deepreg/model/network/cond.py
"""
import tensorflow as tf

from deepreg.model.network.cond import build_conditional_model, conditional_forward
from deepreg.model.network.util import build_backbone


def test_conditional_forward():
    """
    Testing that conditional_forward function returns the tensors with correct shapes
    """

    moving_image_size = (1, 3, 5)
    fixed_image_size = (2, 4, 6)
    batch_size = 1

    local_net = build_backbone(
        image_size=fixed_image_size,
        out_channels=1,
        model_config={
            "backbone": "local",
            "local": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
        },
        method_name="conditional",
    )

    # Check conditional mode network output shapes - Pass
    pred_fixed_label, grid_fixed = conditional_forward(
        backbone=local_net,
        moving_image=tf.ones((batch_size,) + moving_image_size),
        fixed_image=tf.ones((batch_size,) + fixed_image_size),
        moving_label=tf.ones((batch_size,) + moving_image_size),
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
    )
    assert pred_fixed_label.shape == (batch_size,) + fixed_image_size
    assert grid_fixed.shape == fixed_image_size + (3,)


def test_build_conditional_model():
    """
    Testing that build_conditional_model function returns the tensors with correct shapes
    """
    moving_image_size = (1, 3, 5)
    fixed_image_size = (2, 4, 6)
    batch_size = 1

    model = build_conditional_model(
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        index_size=1,
        labeled=True,
        batch_size=batch_size,
        model_config={
            "method": "conditional",
            "backbone": "local",
            "local": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
        },
        loss_config={
            "dissimilarity": {
                "image": {"name": "lncc", "weight": 0.0},
                "label": {
                    "name": "multi_scale",
                    "weight": 1,
                    "multi_scale": {
                        "loss_type": "dice",
                        "loss_scales": [0, 1, 2, 4, 8, 16, 32],
                    },
                },
            },
            "regularization": {"weight": 0.5, "energy_type": "bending"},
        },
    )

    inputs = {
        "moving_image": tf.ones((batch_size,) + moving_image_size),
        "fixed_image": tf.ones((batch_size,) + fixed_image_size),
        "indices": 1,
        "moving_label": tf.ones((batch_size,) + moving_image_size),
        "fixed_label": tf.ones((batch_size,) + fixed_image_size),
    }
    outputs = model(inputs)

    expected_outputs_keys = ["pred_fixed_label"]
    assert all(keys in expected_outputs_keys for keys in outputs)
    assert outputs["pred_fixed_label"].shape == (batch_size,) + fixed_image_size
