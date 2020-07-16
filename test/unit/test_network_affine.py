# coding=utf-8

"""
Tests for deepreg/model/network/affine.py
"""
import tensorflow as tf

from deepreg.model.network.affine import affine_forward, build_affine_model
from deepreg.model.network.util import build_backbone


def test_affine_forward():
    """
    Testing that affine_forward function returns the tensors with correct shapes
    """

    moving_image_size = (1, 3, 5)
    fixed_image_size = (2, 4, 6)
    batch_size = 1

    global_net = build_backbone(
        image_size=fixed_image_size,
        out_channels=3,
        model_config={
            "backbone": "global",
            "global": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
        },
        method_name="affine",
    )

    # Check conditional mode network output shapes - Pass
    affine, ddf, pred_fixed_image, pred_fixed_label, grid_fixed = affine_forward(
        backbone=global_net,
        moving_image=tf.ones((batch_size,) + moving_image_size),
        fixed_image=tf.ones((batch_size,) + fixed_image_size),
        moving_label=tf.ones((batch_size,) + moving_image_size),
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
    )
    assert affine.shape == (batch_size,) + (4,) + (3,)
    assert ddf.shape == (batch_size,) + fixed_image_size + (3,)
    assert pred_fixed_image.shape == (batch_size,) + fixed_image_size
    assert pred_fixed_label.shape == (batch_size,) + fixed_image_size
    assert grid_fixed.shape == fixed_image_size + (3,)


def test_build_affine_model():
    """
    Testing that build_affine_model function returns the tensors with correct shapes
    """
    moving_image_size = (1, 3, 5)
    fixed_image_size = (2, 4, 6)
    batch_size = 1

    model = build_affine_model(
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
        index_size=1,
        labeled=True,
        batch_size=batch_size,
        model_config={
            "method": "affine",
            "backbone": "global",
            "global": {"num_channel_initial": 4, "extract_levels": [1, 2, 3]},
        },
        loss_config={
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

    expected_outputs_keys = ["affine", "ddf", "pred_fixed_label"]
    assert all(keys in expected_outputs_keys for keys in outputs)
    assert outputs["pred_fixed_label"].shape == (batch_size,) + fixed_image_size
    assert outputs["affine"].shape == (batch_size,) + (4,) + (3,)
    assert outputs["ddf"].shape == (batch_size,) + fixed_image_size + (3,)
