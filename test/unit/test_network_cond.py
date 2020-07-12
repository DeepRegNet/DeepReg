# coding=utf-8

"""
Tests for deepreg/model/network/cond.py
"""
import tensorflow as tf

from deepreg.model.network.cond import conditional_forward
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

    # conditional mode
    pred_fixed_label, grid_fixed = conditional_forward(
        backbone=local_net,
        moving_image=tf.random.uniform((batch_size,) + moving_image_size),
        fixed_image=tf.random.uniform((batch_size,) + fixed_image_size),
        moving_label=tf.random.uniform((batch_size,) + moving_image_size),
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
    )
    assert pred_fixed_label.shape == (batch_size,) + fixed_image_size
    assert grid_fixed.shape == fixed_image_size + (3,)
