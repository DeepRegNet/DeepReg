"""
Tests for deepreg/dataset/preprocess.py in
pytest style
"""
import numpy as np

import deepreg.dataset.preprocess as preprocess


def test_transform_no_labels():
    """
    Test transform by comfirming that it transforms
    images and labels only as necessary and when provided.

    Internals of the transform function, such as:
        - layer_util.random_transform_generator
        - layer_util.warp_grid
        - layer_util.resample
    Are assumed working, and are tested separately in
    test_layer_util.py.
    """
    dims = (3, 3, 3)
    batch_size = 2
    moving_image = np.random.uniform(size=dims)
    fixed_image = np.random.uniform(size=dims)

    affine_transform_3d = preprocess.AffineTransformation3D(
        moving_image.shape, fixed_image.shape, batch_size
    )

    moving_image_batched = np.float32(
        np.repeat(moving_image[np.newaxis, :, :, :], batch_size, axis=0)
    )
    fixed_image_batched = np.float32(
        np.repeat(fixed_image[np.newaxis, :, :, :], batch_size, axis=0)
    )
    
    indices = range(batch_size)

    inputs = {
        "moving_image": moving_image_batched,
        "fixed_image": fixed_image_batched,
        "indices": indices,
    }

    outputs = affine_transform_3d.transform(inputs)

    assert outputs.get("moving_image") is not None
    assert outputs.get("fixed_image") is not None
    assert outputs.get("indices") is not None

    assert outputs.get("moving_image").shape == moving_image_batched.shape
    assert outputs.get("fixed_image").shape == fixed_image_batched.shape
    assert len(outputs.get("indices")) == len(indices)


def test_transform_with_labels():
    """
    Test transform by comfirming that it transforms
    images and labels only as necessary and when provided.

     Internals of the transform function, such as:
        - layer_util.random_transform_generator
        - layer_util.warp_grid
        - layer_util.resample
    Are assumed working, and are tested separately in
    test_layer_util.py.
    """
    dims = (3, 3, 3)
    batch_size = 2
    moving_image = np.random.uniform(size=dims)
    fixed_image = np.random.uniform(size=dims)

    affine_transform_3d = preprocess.AffineTransformation3D(
        moving_image.shape, fixed_image.shape, batch_size
    )

    moving_image_batched = np.float32(
        np.repeat(moving_image[np.newaxis, :, :, :], batch_size, axis=0)
    )
    fixed_image_batched = np.float32(
        np.repeat(fixed_image[np.newaxis, :, :, :], batch_size, axis=0)
    )

    moving_label = np.round(np.random.uniform(size=dims))
    fixed_label = np.round(np.random.uniform(size=dims))
    moving_label_batched = np.float32(
        np.repeat(moving_label[np.newaxis, :, :, :], batch_size, axis=0)
    )
    fixed_label_batched = np.float32(
        np.repeat(fixed_label[np.newaxis, :, :, :], batch_size, axis=0)
    )

    indices = range(batch_size)

    inputs = {
        "moving_image": moving_image_batched,
        "fixed_image": fixed_image_batched,
        "moving_label": moving_label_batched,
        "fixed_label": fixed_label_batched,
        "indices": indices,
    }

    outputs = affine_transform_3d.transform(inputs)

    assert outputs.get("moving_image") is not None
    assert outputs.get("fixed_image") is not None
    assert outputs.get("moving_label") is not None
    assert outputs.get("fixed_label") is not None
    assert outputs.get("indices") is not None

    assert outputs.get("moving_image").shape == moving_image_batched.shape
    assert outputs.get("fixed_image").shape == fixed_image_batched.shape
    assert outputs.get("moving_label").shape == moving_image_batched.shape
    assert outputs.get("fixed_label").shape == fixed_image_batched.shape
    assert len(outputs.get("indices")) == len(indices)


def test_preprocess():
    """
    Test preprocess by confirming that it generates
    appropriate datasets for given inputs, and output sizes.
    """
    pass
