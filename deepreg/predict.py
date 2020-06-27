"""
Module to perform predictions on data using
command line interface
"""

import argparse
import logging
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

import deepreg.config.parser as config_parser
import deepreg.dataset.load as load
import deepreg.model.layer_util as layer_util
import deepreg.model.loss.label as label_loss
import deepreg.model.optimizer as opt
from deepreg.model.network.build import build_model

EPS = 1.0e-6


def predict_on_dataset(dataset, fixed_grid_ref, model, save_dir):
    """
    Function to predict results from a dataset from some model
    :param dataset: where data is stored
    :param fixed_grid_ref:
    :param model:
    :param save_dir: str, path to store dir
    """
    metric_map = dict()  # map[image_index][label_index][metric_name] = metric_value
    for _, inputs_dict in enumerate(dataset):
        # pred_fixed_label [batch, f_dim1, f_dim2, f_dim3]
        # moving_image     [batch, m_dim1, m_dim2, m_dim3]
        # fixed_image      [batch, f_dim1, f_dim2, f_dim3]
        # moving_label     [batch, m_dim1, m_dim2, m_dim3]
        # fixed_label      [batch, f_dim1, f_dim2, f_dim3]
        outputs_dict = model.predict(x=inputs_dict)

        moving_image = inputs_dict.get("moving_image")
        fixed_image = inputs_dict.get("fixed_image")
        indices = inputs_dict.get("indices")
        moving_label = inputs_dict.get("moving_label", None)
        fixed_label = inputs_dict.get("fixed_label", None)

        ddf = outputs_dict.get("ddf", None)
        pred_fixed_label = outputs_dict.get("pred_fixed_label", None)

        labeled = moving_label is not None

        num_samples = moving_image.shape[0]
        moving_depth = moving_image.shape[3]
        fixed_depth = fixed_image.shape[3]

        for sample_index in range(num_samples):
            indices_i = indices[sample_index, :].numpy().astype(int).tolist()
            image_index = "_".join([str(x) for x in indices_i[:-1]])
            label_index = str(indices_i[-1])

            # save fixed
            image_dir = os.path.join(save_dir, "image%s" % image_index)
            if labeled:
                image_dir = os.path.join(save_dir, "label%s" % label_index)

            filename_format = os.path.join(
                image_dir, "depth{depth_index:d}_{name:s}.png"
            )
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            for fixed_depth_index in range(fixed_depth):
                fixed_image_d = fixed_image[sample_index, :, :, fixed_depth_index]
                plt.imsave(
                    filename_format.format(
                        depth_index=fixed_depth_index, name="fixed_image"
                    ),
                    fixed_image_d,
                    cmap="gray",
                )  # value range for h5 and nifti might be different

                if labeled:
                    fixed_label_d = fixed_label[sample_index, :, :, fixed_depth_index]
                    fixed_pred_d = pred_fixed_label[
                        sample_index, :, :, fixed_depth_index
                    ]

                    plt.imsave(
                        filename_format.format(
                            depth_index=fixed_depth_index, name="fixed_label"
                        ),
                        fixed_label_d,
                        vmin=0,
                        vmax=1,
                        cmap="gray",
                    )
                    plt.imsave(
                        filename_format.format(
                            depth_index=fixed_depth_index, name="fixed_label_pred"
                        ),
                        fixed_pred_d,
                        vmin=0,
                        vmax=1,
                        cmap="gray",
                    )

            # save moving
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            for moving_depth_index in range(moving_depth):
                moving_image_d = moving_image[sample_index, :, :, moving_depth_index]
                plt.imsave(
                    filename_format.format(
                        depth_index=moving_depth_index, name="moving_image"
                    ),
                    moving_image_d,
                    cmap="gray",
                )  # value range for h5 and nifti might be different
                if labeled:
                    moving_label_d = moving_label[
                        sample_index, :, :, moving_depth_index
                    ]
                    plt.imsave(
                        filename_format.format(
                            depth_index=moving_depth_index, name="moving_label"
                        ),
                        moving_label_d,
                        vmin=0,
                        vmax=1,
                        cmap="gray",
                    )

            # save ddf if exists
            if ddf is not None:
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                for fixed_depth_index in range(fixed_depth):
                    ddf_d = ddf[
                        sample_index, :, :, fixed_depth_index, :
                    ]  # [f_dim1, f_dim2,  3]
                    ddf_max, ddf_min = np.max(ddf_d), np.min(ddf_d)
                    ddf_d = (ddf_d - ddf_min) / np.maximum(ddf_max - ddf_min, EPS)
                    plt.imsave(
                        filename_format.format(
                            depth_index=fixed_depth_index, name="ddf"
                        ),
                        ddf_d,
                    )

            # calculate metric
            if labeled:
                label = fixed_label[sample_index : (sample_index + 1), :, :, :]
                pred = pred_fixed_label[sample_index : (sample_index + 1), :, :, :]
                dice = label_loss.dice_score(y_true=label, y_pred=pred, binary=True)
                dist = label_loss.compute_centroid_distance(
                    y_true=label, y_pred=pred, grid=fixed_grid_ref
                )

                # save metric
                if image_index not in metric_map.keys():
                    metric_map[image_index] = dict()
                # label should not be repeated - assert that it is not in keys
                assert label_index not in metric_map[image_index].keys()
                metric_map[image_index][label_index] = dict(
                    dice=dice.numpy()[0], dist=dist.numpy()[0]
                )

    # print metric
    line_format = (
        "{image_index:s}, label {label_index:s}, dice {dice:.4f}, dist {dist:.4f}\n"
    )
    with open(save_dir + "/metric.log", "w+") as file:
        for image_index in sorted(metric_map.keys()):
            for label_index in sorted(metric_map[image_index].keys()):
                file.write(
                    line_format.format(
                        image_index=image_index,
                        label_index=label_index,
                        **metric_map[image_index][label_index],
                    )
                )


def init(log_dir, ckpt_path):
    """
    Function to create new directory to log directory
    to store results.
    :param log_dir: string, path to store logs.
    :param ckpt_path: str, path where model is stored.
    """
    # check ckpt_path
    if not ckpt_path.endswith(".ckpt"):
        raise ValueError(
            "checkpoint path should end with .ckpt, got {}".format(ckpt_path)
        )

    # init log directory
    log_dir = os.path.join(
        "logs", datetime.now().strftime("%Y%m%d-%H%M%S") if log_dir == "" else log_dir
    )
    if os.path.exists(log_dir):
        logging.warning("Log directory {} exists already.".format(log_dir))
    else:
        os.makedirs(log_dir)

    # load config
    config = config_parser.load("/".join(ckpt_path.split("/")[:-2]) + "/config.yaml")
    return config, log_dir


def predict(gpu, gpu_allow_growth, ckpt_path, mode, batch_size, log_dir, sample_label):
    """
    Function to predict some metrics from the saved model and logging results.
    :param gpu: str, which env gpu to use.
    :param gpu_allow_growth: bool, whether to allow gpu growth or not
    :param ckpt_path: str, where model is stored, should be like
                      log_folder/save/xxx.ckpt
    :param mode: which mode to load the data ??
    :param batch_size: int, batch size to perform predictions in
    :param log_dir: str, path to store logs
    :param sample_label:
    """
    logging.error("TODO sample_label is not used in predict")

    # env vars
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false" if gpu_allow_growth else "true"

    # load config
    config, log_dir = init(log_dir, ckpt_path)
    data_config = config["data"]
    tf_data_config = config["tf"]["data"]
    tf_data_config["batch_size"] = batch_size
    tf_opt_config = config["tf"]["opt"]
    tf_model_config = config["tf"]["model"]
    tf_loss_config = config["tf"]["loss"]

    # data
    data_loader = load.get_data_loader(data_config, mode)
    dataset = data_loader.get_dataset_and_preprocess(
        training=False, repeat=False, **tf_data_config
    )

    # optimizer
    optimizer = opt.get_optimizer(tf_opt_config)

    # model
    model = build_model(
        moving_image_size=data_loader.moving_image_shape,
        fixed_image_size=data_loader.fixed_image_shape,
        index_size=data_loader.num_indices,
        labeled=data_config["labeled"],
        batch_size=tf_data_config["batch_size"],
        tf_model_config=tf_model_config,
        tf_loss_config=tf_loss_config,
    )

    # metrics
    model.compile(optimizer=optimizer)

    # load weights
    # https://stackoverflow.com/questions/58289342/tf2-0-translation-model-error-when-restoring-the-saved-model-unresolved-objec
    model.load_weights(ckpt_path).expect_partial()

    # predict
    fixed_grid_ref = layer_util.get_reference_grid(
        grid_size=data_loader.fixed_image_shape
    )
    predict_on_dataset(
        dataset=dataset,
        fixed_grid_ref=fixed_grid_ref,
        model=model,
        save_dir=log_dir + "/test",
    )


def main(args=None):
    """
    Function to run in command line with argparse to predict results on data
    for a given model
    """
    parser = argparse.ArgumentParser(
        description="predict", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ## ADD POSITIONAL ARGUMENTS
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
        "--log_dir", "-l", help="Path of log directory", default="", type=str
    )

    parser.add_argument(
        "--sample_label",
        "-s",
        help="Method of sampling labels",
        default="all",
        type=str,
    )

    args = parser.parse_args(args)

    predict(
        args.gpu,
        args.gpu_allow_growth,
        args.ckpt_path,
        args.mode,
        args.batch_size,
        args.log_dir,
        args.sample_label,
    )


if __name__ == "__main__":
    main()
