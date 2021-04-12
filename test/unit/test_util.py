import os
import re
import shutil
from test.unit.util import is_equal_np
from typing import Tuple

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
    log_dir = "logs"
    exp_name = "test_build_dataset"
    ckpt_path = ""

    # load config
    config, _, _ = build_config(
        config_path=config_path, log_dir=log_dir, exp_name=exp_name, ckpt_path=ckpt_path
    )

    # build dataset
    data_loader_train, dataset_train, steps_per_epoch_train = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        split="train",
        training=False,
        repeat=False,
    )

    # check output types
    assert isinstance(data_loader_train, DataLoader)
    assert isinstance(dataset_train, tf.data.Dataset)
    assert isinstance(steps_per_epoch_train, int)

    # remove valid data
    config["dataset"]["valid"]["dir"] = ""

    # build dataset
    data_loader_valid, dataset_valid, steps_per_epoch_valid = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        split="valid",
        training=False,
        repeat=False,
    )

    assert data_loader_valid is None
    assert dataset_valid is None
    assert steps_per_epoch_valid is None


@pytest.mark.parametrize("log_dir,exp_name", [("logs", ""), ("logs", "custom")])
def test_build_log_dir(log_dir: str, exp_name: str):
    built_log_dir = build_log_dir(log_dir=log_dir, exp_name=exp_name)
    head, tail = os.path.split(built_log_dir)
    assert head == log_dir
    if exp_name == "":
        # use default timestamp based directory
        pattern = re.compile("[0-9]{8}-[0-9]{6}")
        assert pattern.match(tail)
    else:
        # use custom directory
        assert tail == exp_name


class TestSaveArray:
    save_dir = "logs/test_util_save_array"
    arr_name = "arr"
    png_dir = os.path.join(save_dir, arr_name)
    dim_err_msg = "arr must be 3d or 4d numpy array or tf tensor"
    ch_err_msg = "4d arr must have 3 channels as last dimension"

    def setup_method(self, method):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    def teardown_method(self, method):
        if os.path.exists(self.save_dir):
            shutil.rmtree(self.save_dir)

    @staticmethod
    def get_num_files_in_dir(dir_path: str, suffix: str):
        if os.path.exists(dir_path):
            return len([x for x in os.listdir(dir_path) if x.endswith(suffix)])
        return 0

    @pytest.mark.parametrize(
        "arr",
        [
            tf.random.uniform(shape=(2, 3, 4)),
            tf.random.uniform(shape=(2, 3, 4, 3)),
            np.random.rand(2, 3, 4),
            np.random.rand(2, 3, 4, 3),
        ],
    )
    def test_3d_4d(self, arr: Tuple[tf.Tensor, np.ndarray]):
        save_array(save_dir=self.save_dir, arr=arr, name=self.arr_name, normalize=True)
        assert self.get_num_files_in_dir(self.png_dir, suffix=".png") == 4
        assert self.get_num_files_in_dir(self.save_dir, suffix=".nii.gz") == 1

    @pytest.mark.parametrize(
        "arr,err_msg",
        [
            [tf.random.uniform(shape=(2, 3, 4, 3, 3)), dim_err_msg],
            [tf.random.uniform(shape=(2, 3, 4, 1)), ch_err_msg],
            [np.random.rand(2, 3, 4, 3, 3), dim_err_msg],
            [np.random.rand(2, 3, 4, 1), ch_err_msg],
        ],
    )
    def test_wrong_shape(self, arr: Tuple[tf.Tensor, np.ndarray], err_msg: str):
        with pytest.raises(ValueError) as err_info:
            save_array(
                save_dir=self.save_dir, arr=arr, name=self.arr_name, normalize=True
            )
        assert err_msg in str(err_info.value)

    @pytest.mark.parametrize("save_nifti", [True, False])
    def test_save_nifti(self, save_nifti: bool):
        arr = np.random.rand(2, 3, 4, 3)
        save_array(
            save_dir=self.save_dir,
            arr=arr,
            name=self.arr_name,
            normalize=True,
            save_nifti=save_nifti,
        )
        assert self.get_num_files_in_dir(self.save_dir, suffix=".nii.gz") == int(
            save_nifti
        )

    @pytest.mark.parametrize("save_png", [True, False])
    def test_save_png(self, save_png: bool):
        arr = np.random.rand(2, 3, 4, 3)
        save_array(
            save_dir=self.save_dir,
            arr=arr,
            name=self.arr_name,
            normalize=True,
            save_png=save_png,
        )
        assert (
            self.get_num_files_in_dir(self.png_dir, suffix=".png") == int(save_png) * 4
        )

    @pytest.mark.parametrize("overwrite", [True, False])
    def test_overwrite(self, overwrite: bool):
        arr1 = np.random.rand(2, 3, 4, 3)
        arr2 = arr1 + 1
        nifti_file_path = os.path.join(self.save_dir, self.arr_name + ".nii.gz")
        # save arr1
        os.makedirs(self.save_dir, exist_ok=True)
        nib.save(img=nib.Nifti1Image(arr1, affine=np.eye(4)), filename=nifti_file_path)
        # save arr2 w/o overwrite
        save_array(
            save_dir=self.save_dir,
            arr=arr2,
            name=self.arr_name,
            normalize=True,
            overwrite=overwrite,
        )
        arr_read = load_nifti_file(file_path=nifti_file_path)
        assert is_equal_np(arr2 if overwrite else arr1, arr_read)


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
