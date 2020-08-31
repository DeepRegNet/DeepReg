import os
import re
import shutil
from test.unit.util import is_equal_np

import nibabel as nib
import numpy as np
import pytest
import tensorflow as tf

from deepreg.dataset.loader.interface import DataLoader
from deepreg.dataset.loader.nifti_loader import load_nifti_file
from deepreg.train import build_config
from deepreg.util import (
    build_dataset,
    build_log_dir,
    calculate_metrics,
    save_array,
    save_metric_dict,
)


def test_build_dataset():
    """
    Test build_dataset by checking the output types
    """

    # init arguments
    config_path = "config/unpaired_labeled_ddf.yaml"
    log_dir = "test_build_dataset"
    ckpt_path = ""

    # load config
    config, log_dir = build_config(
        config_path=config_path, log_dir=log_dir, ckpt_path=ckpt_path
    )

    # build dataset
    data_loader_train, dataset_train, steps_per_epoch_train = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        mode="train",
        training=False,
        repeat=False,
    )

    # check output types
    assert isinstance(data_loader_train, DataLoader)
    assert isinstance(dataset_train, tf.data.Dataset)
    assert isinstance(steps_per_epoch_train, int)

    # remove valid data
    config["dataset"]["dir"]["valid"] = ""

    # build dataset
    data_loader_valid, dataset_valid, steps_per_epoch_valid = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        mode="valid",
        training=False,
        repeat=False,
    )

    assert data_loader_valid is None
    assert dataset_valid is None
    assert steps_per_epoch_valid is None


def test_build_log_dir():
    """
    Test build_log_dir for default directory and custom directory
    """

    # use default timestamp based directory
    log_dir = build_log_dir(log_dir="")
    head, tail = os.path.split(log_dir)
    assert head == "logs"
    pattern = re.compile("[0-9]{8}-[0-9]{6}")
    assert pattern.match(tail)

    # use custom directory
    log_dir = build_log_dir(log_dir="custom")
    head, tail = os.path.split(log_dir)
    assert head == "logs"
    assert tail == "custom"


def test_save_array():
    """
    Test save_array by testing different shapes and count output files
    """

    def get_num_pngs_in_dir(dir_paths):
        return len([x for x in os.listdir(dir_paths) if x.endswith(".png")])

    def get_num_niftis_in_dir(dir_paths):
        return len([x for x in os.listdir(dir_paths) if x.endswith(".nii.gz")])

    save_dir = "logs/test_util_save_array"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # test 3D tf tensor
    name = "3d_tf"
    out_dir = os.path.join(save_dir, name)
    arr = tf.random.uniform(shape=(2, 3, 4))
    save_array(save_dir=save_dir, arr=arr, name=name, gray=True)
    assert get_num_pngs_in_dir(out_dir) == 4
    assert get_num_niftis_in_dir(save_dir) == 1
    shutil.rmtree(out_dir)
    os.remove(os.path.join(save_dir, name + ".nii.gz"))

    # test 4D tf tensor
    name = "4d_tf"
    out_dir = os.path.join(save_dir, name)
    arr = tf.random.uniform(shape=(2, 3, 4, 3))
    save_array(save_dir=save_dir, arr=arr, name=name, gray=True)
    assert get_num_pngs_in_dir(out_dir) == 4
    assert get_num_niftis_in_dir(save_dir) == 1
    shutil.rmtree(out_dir)
    os.remove(os.path.join(save_dir, name + ".nii.gz"))

    # test 3D np tensor
    name = "3d_np"
    out_dir = os.path.join(save_dir, name)
    arr = np.random.rand(2, 3, 4)
    save_array(save_dir=save_dir, arr=arr, name=name, gray=True)
    assert get_num_pngs_in_dir(out_dir) == 4
    assert get_num_niftis_in_dir(save_dir) == 1
    shutil.rmtree(out_dir)
    os.remove(os.path.join(save_dir, name + ".nii.gz"))

    # test 4D np tensor
    name = "4d_np"
    out_dir = os.path.join(save_dir, name)
    arr = np.random.rand(2, 3, 4, 3)
    save_array(save_dir=save_dir, arr=arr, name=name, gray=True)
    assert get_num_pngs_in_dir(out_dir) == 4
    assert get_num_niftis_in_dir(save_dir) == 1
    shutil.rmtree(out_dir)
    os.remove(os.path.join(save_dir, name + ".nii.gz"))

    # test 4D np tensor without nifti
    name = "4d_np"
    out_dir = os.path.join(save_dir, name)
    arr = np.random.rand(2, 3, 4, 3)
    save_array(save_dir=save_dir, arr=arr, name=name, gray=True, save_nifti=False)
    assert get_num_pngs_in_dir(out_dir) == 4
    assert get_num_niftis_in_dir(save_dir) == 0
    shutil.rmtree(out_dir)

    # test 4D np tensor without png
    name = "4d_np"
    out_dir = os.path.join(save_dir, name)
    arr = np.random.rand(2, 3, 4, 3)
    assert not os.path.exists(out_dir)
    save_array(save_dir=save_dir, arr=arr, name=name, gray=True, save_png=False)
    assert not os.path.exists(out_dir)
    assert get_num_niftis_in_dir(save_dir) == 1
    os.remove(os.path.join(save_dir, name + ".nii.gz"))

    # test 4D np tensor with overwrite
    name = "4d_np"
    out_dir = os.path.join(save_dir, name)
    arr1 = np.random.rand(2, 3, 4, 3)
    arr2 = np.random.rand(2, 3, 4, 3)
    assert not is_equal_np(arr1, arr2)
    nifti_file_path = os.path.join(save_dir, name + ".nii.gz")
    # save arr1
    os.makedirs(save_dir, exist_ok=True)
    nib.save(img=nib.Nifti2Image(arr1, affine=np.eye(4)), filename=nifti_file_path)
    # save arr2 without overwrite
    save_array(save_dir=save_dir, arr=arr1, name=name, gray=True, overwrite=False)
    arr_read = load_nifti_file(file_path=nifti_file_path)
    assert is_equal_np(arr1, arr_read)
    # save arr2 with overwrite
    save_array(save_dir=save_dir, arr=arr2, name=name, gray=True, overwrite=True)
    arr_read = load_nifti_file(file_path=nifti_file_path)
    assert is_equal_np(arr2, arr_read)
    shutil.rmtree(out_dir)
    os.remove(os.path.join(save_dir, name + ".nii.gz"))

    # test 5D np tensor
    name = "5d_np"
    arr = np.random.rand(2, 3, 4, 1, 3)
    with pytest.raises(ValueError) as err_info:
        save_array(save_dir=save_dir, arr=arr, name=name, gray=True)
    assert "arr must be 3d or 4d numpy array or tf tensor" in str(err_info.value)

    # test 4D np tensor with wrong shape
    name = "5d_np"
    arr = np.random.rand(2, 3, 4, 1)
    with pytest.raises(ValueError) as err_info:
        save_array(save_dir=save_dir, arr=arr, name=name, gray=True)
    assert "4d arr must have 3 channels as last dimension" in str(err_info.value)


def test_calculate_metrics():
    """
    Test calculate_metrics by checking output keys.
    Assuming the metrics functions are correct.
    """

    batch_size = 2
    fixed_image_shape = (4, 4, 4)  # (f_dim1, f_dim2, f_dim3)

    fixed_image = tf.random.uniform(shape=(batch_size,) + fixed_image_shape)
    fixed_label = tf.random.uniform(shape=(batch_size,) + fixed_image_shape)
    pred_fixed_image = tf.random.uniform(shape=(batch_size,) + fixed_image_shape)
    pred_fixed_label = tf.random.uniform(shape=(batch_size,) + fixed_image_shape)
    fixed_grid_ref = tf.random.uniform(shape=(1,) + fixed_image_shape + (3,))
    sample_index = 0

    # labeled and have pred_fixed_image
    got = calculate_metrics(
        fixed_image=fixed_image,
        fixed_label=fixed_label,
        pred_fixed_image=pred_fixed_image,
        pred_fixed_label=pred_fixed_label,
        fixed_grid_ref=fixed_grid_ref,
        sample_index=sample_index,
    )
    assert got["image_ssd"] is not None
    assert got["label_binary_dice"] is not None
    assert got["label_tre"] is not None
    assert sorted(list(got.keys())) == sorted(
        ["image_ssd", "label_binary_dice", "label_tre"]
    )

    # labeled and do not have pred_fixed_image
    got = calculate_metrics(
        fixed_image=fixed_image,
        fixed_label=fixed_label,
        pred_fixed_image=None,
        pred_fixed_label=pred_fixed_label,
        fixed_grid_ref=fixed_grid_ref,
        sample_index=sample_index,
    )
    assert got["image_ssd"] is None
    assert got["label_binary_dice"] is not None
    assert got["label_tre"] is not None

    # unlabeled and have pred_fixed_image
    got = calculate_metrics(
        fixed_image=fixed_image,
        fixed_label=None,
        pred_fixed_image=pred_fixed_image,
        pred_fixed_label=None,
        fixed_grid_ref=fixed_grid_ref,
        sample_index=sample_index,
    )
    assert got["image_ssd"] is not None
    assert got["label_binary_dice"] is None
    assert got["label_tre"] is None

    # unlabeled and do not have pred_fixed_image
    got = calculate_metrics(
        fixed_image=fixed_image,
        fixed_label=None,
        pred_fixed_image=None,
        pred_fixed_label=None,
        fixed_grid_ref=fixed_grid_ref,
        sample_index=sample_index,
    )
    assert got["image_ssd"] is None
    assert got["label_binary_dice"] is None
    assert got["label_tre"] is None


def test_save_metric_dict():
    """
    Test save_metric_dict by checking output files.
    """

    save_dir = "logs/test_save_metric_dict"
    metrics = [
        dict(image_ssd=0.1, label_dice=0.8, pair_index=[0], label_index=0),
        dict(image_ssd=0.2, label_dice=0.7, pair_index=[1], label_index=1),
        dict(image_ssd=0.3, label_dice=0.6, pair_index=[2], label_index=0),
    ]
    save_metric_dict(save_dir=save_dir, metrics=metrics)
    assert len([x for x in os.listdir(save_dir) if x.endswith(".csv")]) == 3
    shutil.rmtree(save_dir)
