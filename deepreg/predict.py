# coding=utf-8

"""
Module to perform predictions on data using
command line interface.
"""

import argparse
import logging
import os
import shutil

import numpy as np
import tensorflow as tf

import deepreg.model.layer_util as layer_util
import deepreg.model.optimizer as opt
import deepreg.parser as config_parser
from deepreg.model.network.build import build_model
from deepreg.util import (
    build_dataset,
    build_log_dir,
    calculate_metrics,
    save_array,
    save_metric_dict,
)


def build_pair_output_path(indices: list, save_dir: str) -> (str, str):
    """
    Create directory for saving the paired data

    :param indices: indices of the pair, the last one is for label
    :param save_dir: directory of output
    :return: - save_dir, str, directory for saving the moving/fixed image
             - label_dir, str, directory for saving the rest outputs
    """

    # cast indices to string and init directory name
    pair_index = "pair_" + "_".join([str(x) for x in indices[:-1]])
    pair_dir = os.path.join(save_dir, pair_index)
    os.makedirs(pair_dir, exist_ok=True)

    if indices[-1] >= 0:
        label_index = f"label_{indices[-1]}"
        label_dir = os.path.join(pair_dir, label_index)
        os.makedirs(label_dir, exist_ok=True)
    else:
        label_dir = pair_dir

    return pair_dir, label_dir


def predict_on_dataset(
    dataset: tf.data.Dataset,
    fixed_grid_ref: tf.Tensor,
    model: tf.keras.Model,
    model_method: str,
    save_dir: str,
    save_nifti: bool,
    save_png: bool,
):
    """
    Function to predict results from a dataset from some model

    :param dataset: where data is stored
    :param fixed_grid_ref: shape=(1, f_dim1, f_dim2, f_dim3, 3)
    :param model: model to be used for prediction
    :param model_method: str, ddf / dvf / affine / conditional
    :param save_dir: str, path to store dir
    :param save_nifti: if true, outputs will be saved in nifti format
    :param save_png: if true, outputs will be saved in png format
    """
    # remove the save_dir in case it exists
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    sample_index_strs = []
    metric_lists = []
    for _, inputs_dict in enumerate(dataset):
        batch_size = inputs_dict[list(inputs_dict.keys())[0]].shape[0]
        outputs_dict = model.predict(x=inputs_dict, batch_size=batch_size)

        # moving image/label
        # (batch, m_dim1, m_dim2, m_dim3)
        moving_image = inputs_dict["moving_image"]
        moving_label = inputs_dict.get("moving_label", None)
        # fixed image/labelimage_index
        # (batch, f_dim1, f_dim2, f_dim3)
        fixed_image = inputs_dict["fixed_image"]
        fixed_label = inputs_dict.get("fixed_label", None)

        # indices to identify the pair
        # (batch, num_indices) last indice is for label, -1 means unlabeled data
        indices = inputs_dict.get("indices")
        # ddf / dvf
        # (batch, f_dim1, f_dim2, f_dim3, 3)
        ddf = outputs_dict.get("ddf", None)
        dvf = outputs_dict.get("dvf", None)
        affine = outputs_dict.get("affine", None)  # (batch, 4, 3)

        # prediction
        # (batch, f_dim1, f_dim2, f_dim3)
        pred_fixed_label = outputs_dict.get("pred_fixed_label", None)
        pred_fixed_image = (
            layer_util.resample(vol=moving_image, loc=fixed_grid_ref + ddf)
            if ddf is not None
            else None
        )

        # save images of inputs and outputs
        for sample_index in range(moving_image.shape[0]):
            # save moving/fixed image under pair_dir
            # save moving/fixed label, pred fixed image/label, ddf/dvf under label dir
            # if labeled, label dir is a sub dir of pair_dir, otherwise = pair_dir

            # init output path
            indices_i = indices[sample_index, :].numpy().astype(int).tolist()
            pair_dir, label_dir = build_pair_output_path(
                indices=indices_i, save_dir=save_dir
            )

            # save image/label
            # if model is conditional, the pred_fixed_image depends on the input label
            conditional = model_method == "conditional"
            arr_save_dirs = [
                pair_dir,
                pair_dir,
                label_dir if conditional else pair_dir,
                label_dir,
                label_dir,
                label_dir,
            ]
            arrs = [
                moving_image,
                fixed_image,
                pred_fixed_image,
                moving_label,
                fixed_label,
                pred_fixed_label,
            ]
            names = [
                "moving_image",
                "fixed_image",
                "pred_fixed_image",  # or warped moving image
                "moving_label",
                "fixed_label",
                "pred_fixed_label",  # or warped moving label
            ]
            for arr_save_dir, arr, name in zip(arr_save_dirs, arrs, names):
                if arr is not None:
                    # for files under pair_dir, do not overwrite
                    save_array(
                        save_dir=arr_save_dir,
                        arr=arr[sample_index, :, :, :],
                        name=name,
                        normalize="image" in name,  # label's value is already in [0, 1]
                        save_nifti=save_nifti,
                        save_png=save_png,
                        overwrite=arr_save_dir == label_dir,
                    )

            # save ddf / dvf
            arrs = [ddf, dvf]
            names = ["ddf", "dvf"]
            for arr, name in zip(arrs, names):
                if arr is not None:
                    save_array(
                        save_dir=label_dir if conditional else pair_dir,
                        arr=arr[sample_index, :, :, :],
                        name=name,
                        normalize=True,
                        save_nifti=save_nifti,
                        save_png=save_png,
                    )

            # save affine
            if affine is not None:
                np.savetxt(
                    fname=os.path.join(
                        label_dir if conditional else pair_dir, "affine.txt"
                    ),
                    x=affine[sample_index, :, :].numpy(),
                    delimiter=",",
                )

            # calculate metric
            sample_index_str = "_".join([str(x) for x in indices_i])
            if sample_index_str in sample_index_strs:
                raise ValueError(
                    "Sample is repeated, maybe the dataset has been repeated."
                )
            sample_index_strs.append(sample_index_str)

            metric = calculate_metrics(
                fixed_image=fixed_image,
                fixed_label=fixed_label,
                pred_fixed_image=pred_fixed_image,
                pred_fixed_label=pred_fixed_label,
                fixed_grid_ref=fixed_grid_ref,
                sample_index=sample_index,
            )
            metric["pair_index"] = indices_i[:-1]
            metric["label_index"] = indices_i[-1]
            metric_lists.append(metric)

    # save metric
    save_metric_dict(save_dir=save_dir, metrics=metric_lists)


def build_config(
    config_path: (str, list), log_root: str, log_dir: str, ckpt_path: str
) -> [dict, str]:
    """
    Function to create new directory to log directory to store results.

    :param config_path: string or list of strings, path of configuration files
    :param log_root: str, root of logs
    :param log_dir: string, path to store logs.
    :param ckpt_path: str, path where model is stored.
    :return: - config, configuration dictionary
             - log_dir, path of the directory for saving outputs
    """
    # check ckpt_path
    if not ckpt_path.endswith(".ckpt"):
        raise ValueError(
            "checkpoint path should end with .ckpt, got {}".format(ckpt_path)
        )

    # init log directory
    log_dir = build_log_dir(log_root=log_root, log_dir=log_dir)

    # load config
    if config_path == "":
        # use default config, which should be provided in the log folder
        config = config_parser.load_configs(
            "/".join(ckpt_path.split("/")[:-2]) + "/config.yaml"
        )
    else:
        # use customized config
        logging.warning(
            "Using customized configuration."
            "The code might break if the config of the model doesn't match the saved model."
        )
        config = config_parser.load_configs(config_path)
    return config, log_dir


def predict(
    gpu: str,
    gpu_allow_growth: bool,
    ckpt_path: str,
    mode: str,
    batch_size: int,
    log_dir: str,
    sample_label: str,
    config_path: (str, list),
    save_nifti: bool = True,
    save_png: bool = True,
    log_root: str = "logs",
):
    """
    Function to predict some metrics from the saved model and logging results.

    :param gpu: str, which env gpu to use.
    :param gpu_allow_growth: bool, whether to allow gpu growth or not
    :param ckpt_path: str, where model is stored, should be like log_folder/save/xxx.ckpt
    :param mode: train / valid / test, to define which split of dataset to be evaluated
    :param batch_size: int, batch size to perform predictions in
    :param log_dir: str, path to store logs
    :param sample_label: sample/all, not used
    :param save_nifti: if true, outputs will be saved in nifti format
    :param save_png: if true, outputs will be saved in png format
    :param config_path: to overwrite the default config
    """
    # TODO support custom sample_label
    logging.warning(
        "sample_label is not used in predict. It is True if and only if mode == 'train'."
    )

    # env vars
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false" if gpu_allow_growth else "true"

    # load config
    config, log_dir = build_config(
        config_path=config_path, log_root=log_root, log_dir=log_dir, ckpt_path=ckpt_path
    )
    preprocess_config = config["train"]["preprocess"]
    # batch_size corresponds to batch_size per GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    preprocess_config["batch_size"] = batch_size * max(len(gpus), 1)

    # data
    data_loader, dataset, _ = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=preprocess_config,
        mode=mode,
        training=False,
        repeat=False,
    )

    # optimizer
    optimizer = opt.build_optimizer(optimizer_config=config["train"]["optimizer"])

    # model
    model = build_model(
        moving_image_size=data_loader.moving_image_shape,
        fixed_image_size=data_loader.fixed_image_shape,
        index_size=data_loader.num_indices,
        labeled=config["dataset"]["labeled"],
        batch_size=preprocess_config["batch_size"],
        model_config=config["train"]["model"],
        loss_config=config["train"]["loss"],
    )

    # metrics
    model.compile(optimizer=optimizer)

    # load weights
    # https://stackoverflow.com/questions/58289342/tf2-0-translation-model-error-when-restoring-the-saved-model-unresolved-objec
    model.load_weights(ckpt_path).expect_partial()

    # predict
    fixed_grid_ref = tf.expand_dims(
        layer_util.get_reference_grid(grid_size=data_loader.fixed_image_shape), axis=0
    )  # shape = (1, f_dim1, f_dim2, f_dim3, 3)
    predict_on_dataset(
        dataset=dataset,
        fixed_grid_ref=fixed_grid_ref,
        model=model,
        model_method=config["train"]["model"]["method"],
        save_dir=log_dir + "/test",
        save_nifti=save_nifti,
        save_png=save_png,
    )

    # close the opened files in data loaders
    data_loader.close()


def main(args=None):
    """
    Function to run in command line with argparse to predict results on data
    for a given model.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        "-g",
        help="GPU index for training."
        '-g "" for using CPU'
        '-g "0" for using GPU 0'
        '-g "0,1" for using GPU 0 and 1.',
        type=str,
        required=True,
    )

    parser.add_argument(
        "--gpu_allow_growth",
        "-gr",
        help="Prevent TensorFlow from reserving all available GPU memory",
        default=False,
    )

    parser.add_argument(
        "--ckpt_path",
        "-k",
        help="Path of checkpointed model to load",
        default="",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--mode",
        "-m",
        help="Define the split of data to be used for prediction. One of train / valid / test",
        type=str,
        default="test",
        required=True,
    )

    parser.add_argument(
        "--batch_size", "-b", help="Batch size for predictions", default=1, type=int
    )

    parser.add_argument(
        "--log_root", help="Root of log directory.", default="logs", type=str
    )

    parser.add_argument(
        "--log_dir", "-l", help="Path of log directory", default="", type=str
    )

    # TODO use this argument
    parser.add_argument(
        "--sample_label",
        "-s",
        help="Method of sampling labels",
        default="all",
        type=str,
    )

    parser.add_argument("--save_nifti", dest="nifti", action="store_true")
    parser.add_argument("--no_nifti", dest="nifti", action="store_false")
    parser.set_defaults(nifti=True)

    parser.add_argument("--save_png", dest="png", action="store_true")
    parser.add_argument("--no_png", dest="png", action="store_false")
    parser.set_defaults(png=False)

    parser.add_argument(
        "--config_path",
        "-c",
        help="Path of config, must end with .yaml. Can pass multiple paths.",
        type=str,
        nargs="*",
        default="",
    )

    args = parser.parse_args(args)

    predict(
        gpu=args.gpu,
        gpu_allow_growth=args.gpu_allow_growth,
        ckpt_path=args.ckpt_path,
        mode=args.mode,
        batch_size=args.batch_size,
        log_root=args.log_root,
        log_dir=args.log_dir,
        sample_label=args.sample_label,
        config_path=args.config_path,
        save_nifti=args.nifti,
        save_png=args.png,
    )


if __name__ == "__main__":
    main()
