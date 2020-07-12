import tensorflow as tf

import deepreg.dataset.preprocess as preprocess


def test_resize_inputs():
    """
   Test _gen_transforms by confirming that it generates
   appropriate output sizes.
   """
    input_size = (2, 2, 2)
    moving_image_size = (1, 3, 5)
    fixed_image_size = (2, 4, 6)

    # labeled data - Pass
    moving_image = tf.ones(input_size)
    fixed_image = tf.ones(input_size)
    moving_label = tf.ones(input_size)
    fixed_label = tf.ones(input_size)
    indices = tf.ones((2,))

    inputs = dict(
        moving_image=moving_image,
        fixed_image=fixed_image,
        moving_label=moving_label,
        fixed_label=fixed_label,
        indices=indices,
    )
    outputs = preprocess.resize_inputs(
        inputs=inputs,
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
    )

    assert outputs["moving_image"].shape == moving_image_size
    assert outputs["fixed_image"].shape == fixed_image_size
    assert outputs["moving_label"].shape == moving_image_size
    assert outputs["fixed_label"].shape == fixed_image_size

    # unlabeled data - Pass
    moving_image = tf.ones(input_size)
    fixed_image = tf.ones(input_size)
    indices = tf.ones((2,))

    inputs = dict(moving_image=moving_image, fixed_image=fixed_image, indices=indices)
    outputs = preprocess.resize_inputs(
        inputs=inputs,
        moving_image_size=moving_image_size,
        fixed_image_size=fixed_image_size,
    )

    assert outputs["moving_image"].shape == moving_image_size
    assert outputs["fixed_image"].shape == fixed_image_size
