"""
Module to perform predictions on data using
command line interface
"""

import argparse
import logging
import os

import numpy as np
import tensorflow as tf

import deepreg.config.parser as config_parser
import deepreg.model.layer_util as layer_util
import deepreg.model.loss.label as label_loss
import deepreg.model.optimizer as opt
from deepreg.model.network.build import build_model
from deepreg.util import build_dataset, build_log_dir, get_mean_median_std, save_array

# import deepreg.model.loss.image as image_loss
EPS = 1.0e-6
OUT_FILE_PATH_FORMAT = os.path.join(
    "{sample_dir:s}", "depth{depth_index:d}_{name:s}.png"
)


def build_sample_output_path(
    indices: list, save_dir: str, labeled: bool
) -> [str, str, str]:
    """
    Create directory for saving the sample
    :param indices: indices of the sample, assuming the last one is for label
    :param save_dir: directory of output
    :param labeled: if data are labeled
    :return:
    - image_index, str, string to identify the image
    - label_index, str, string to identify the label
    - sample_dir, str, directory for saving the directory of the sample
    """

    # cast indices to string
    image_index = "_".join([str(x) for x in indices[:-1]])
    label_index = str(indices[-1])

    # init directory name
    sample_dir = os.path.join(save_dir, f"image{image_index}")
    if labeled:
        # add label to the output path
        sample_dir = os.path.join(sample_dir, f"label{label_index}")

    # create the directory of the path
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    return image_index, label_index, sample_dir


def predict_on_dataset(dataset, fixed_grid_ref, model, save_dir):
    """
    Function to predict results from a dataset from some model
    :param dataset: where data is stored
    :param fixed_grid_ref: (1, f_dim1, f_dim2, f_dim3, 3)
    :param model:
    :param save_dir: str, path to store dir
    """
    # image_metric_map = dict()  # map[image_index][metric_name] = metric_value
    label_metric_map = (
        dict()
    )  # map[image_index][label_index][metric_name] = metric_value

    for _, inputs_dict in enumerate(dataset):
        outputs_dict = model.predict(x=inputs_dict)

        moving_image = inputs_dict.get(
            "moving_image"
        )  # (batch, m_dim1, m_dim2, m_dim3)
        fixed_image = inputs_dict.get("fixed_image")  # (batch, f_dim1, f_dim2, f_dim3)
        moving_label = inputs_dict.get(
            "moving_label", None
        )  # (batch, m_dim1, m_dim2, m_dim3)
        fixed_label = inputs_dict.get(
            "fixed_label", None
        )  # (batch, f_dim1, f_dim2, f_dim3)
        indices = inputs_dict.get(
            "indices"
        )  # (batch, num_indices) last indice is for label if labeled
        labeled = moving_label is not None

        ddf = outputs_dict.get("ddf", None)  # (batch, f_dim1, f_dim2, f_dim3, 3)
        dvf = outputs_dict.get("dvf", None)  # (batch, f_dim1, f_dim2, f_dim3, 3)
        # affine = outputs_dict.get("affine", None) # (batch, 4, 3)
        pred_fixed_label = outputs_dict.get(
            "pred_fixed_label", None
        )  # (batch, f_dim1, f_dim2, f_dim3)
        # moving_image_warped = (
        #     layer_util.resample(vol=moving_image, loc=fixed_grid_ref + ddf)
        #     if ddf is not None
        #     else None
        # )  # (batch, f_dim1, f_dim2, f_dim3)

        # save images of inputs and outputs
        for sample_index in range(moving_image.shape[0]):
            # init output path
            indices_i = indices[sample_index, :].numpy().astype(int).tolist()
            image_index, label_index, sample_dir = build_sample_output_path(
                indices=indices_i, save_dir=save_dir, labeled=labeled
            )

            # save image
            save_array(
                sample_dir=sample_dir,
                arr=fixed_image[sample_index, :, :, :],
                prefix="fixed",
                name="image",
                gray=True,
            )
            save_array(
                sample_dir=sample_dir,
                arr=moving_image[sample_index, :, :, :],
                prefix="moving",
                name="image",
                gray=True,
            )
            # if moving_image_warped is not None:
            #     save_array(
            #         sample_dir=sample_dir,
            #         arr=moving_image_warped[sample_index, :, :, :],
            #         prefix="fixed",
            #         name="warped_image",
            #         gray=True,
            #     )

            # save label
            if labeled:
                save_array(
                    sample_dir=sample_dir,
                    arr=fixed_label[sample_index, :, :, :],
                    prefix="fixed",
                    name="label",
                    gray=True,
                )
                save_array(
                    sample_dir=sample_dir,
                    arr=moving_label[sample_index, :, :, :],
                    prefix="moving",
                    name="label",
                    gray=True,
                )
                save_array(
                    sample_dir=sample_dir,
                    arr=pred_fixed_label[sample_index, :, :, :],
                    prefix="fixed",
                    name="label_pred",
                    gray=True,
                )

            # save ddf / dvf if exists
            for field, field_name in zip([ddf, dvf], ["ddf", "dvf"]):
                if field is not None:
                    # normalize field values into 0-1
                    arr = field[sample_index, :, :, :, :]
                    field_max, field_min = np.max(arr), np.min(arr)
                    arr = (arr - field_min) / np.maximum(field_max - field_min, EPS)
                    save_array(
                        sample_dir=sample_dir,
                        arr=arr,
                        prefix="fixed",
                        name=field_name,
                        gray=False,
                    )
                    save_array(
                        sample_dir=sample_dir,
                        arr=arr,
                        prefix="fixed",
                        name=field_name + "_gray",
                        gray=True,
                    )

            # calculate metric
            # TODO buggy code
            # if moving_image_warped is not None:
            #     if image_index in image_metric_map.keys():
            #         raise ValueError(
            #             "Image index is repeated, maybe the dataset has been repeated."
            #         )
            #     y_true = fixed_image[sample_index : (sample_index + 1), :, :, :]
            #     y_pred = moving_image_warped[sample_index : (sample_index + 1), :, :, :]
            #     ssd = image_loss.ssd(y_true=y_true, y_pred=y_pred).numpy()[0]
            #     image_metric_map[image_index] = dict(ssd=ssd)

            if labeled:
                y_true = fixed_label[sample_index : (sample_index + 1), :, :, :]
                y_pred = pred_fixed_label[sample_index : (sample_index + 1), :, :, :]
                dice = label_loss.dice_score(y_true=y_true, y_pred=y_pred, binary=True)
                tre = label_loss.compute_centroid_distance(
                    y_true=y_true, y_pred=y_pred, grid=fixed_grid_ref
                )

                if image_index not in label_metric_map.keys():
                    label_metric_map[image_index] = dict()
                if label_index in label_metric_map[image_index].keys():
                    raise ValueError(
                        "Label index is repeated, maybe the dataset has been repeated."
                    )
                label_metric_map[image_index][label_index] = dict(
                    dice=dice.numpy()[0], tre=tre.numpy()[0]
                )

    # save metric
    with open(save_dir + "/label_metric.log", "w+") as file:
        # save details of each image and label
        label_metric_per_label = dict(dice={}, tre={})
        for image_index in sorted(label_metric_map.keys()):
            for label_index in sorted(label_metric_map[image_index].keys()):
                dice = label_metric_map[image_index][label_index]["dice"]
                tre = label_metric_map[image_index][label_index]["tre"]
                if label_index not in label_metric_per_label["dice"]:
                    label_metric_per_label["dice"][label_index] = []
                if label_index not in label_metric_per_label["tre"]:
                    label_metric_per_label["tre"][label_index] = []
                label_metric_per_label["dice"][label_index].append(dice)
                label_metric_per_label["tre"][label_index].append(tre)
                file.write(
                    f"{image_index}, label {label_index}, dice {dice}, tre {tre}\n"
                )
        # save stats on each label
        file.write("\n\n")
        for label_index in sorted(label_metric_per_label["dice"].keys()):
            dice_mean, dice_median, dice_std = get_mean_median_std(
                label_metric_per_label["dice"][label_index]
            )
            tre_mean, tre_median, tre_std = get_mean_median_std(
                label_metric_per_label["tre"][label_index]
            )
            file.write(
                f"label {label_index}, "
                f"dice mean={dice_mean}, median={dice_median}, std={dice_std}, "
                f"tre mean={tre_mean}, median={tre_median}, std={tre_std}, \n"
            )
        # save stats overall
        file.write("\n\n")
        dices = [v for vs in label_metric_per_label["dice"].values() for v in vs]
        tres = [v for vs in label_metric_per_label["tre"].values() for v in vs]
        dice_mean, dice_median, dice_std = get_mean_median_std(dices)
        tre_mean, tre_median, tre_std = get_mean_median_std(tres)
        file.write(
            f"All, "
            f"dice mean={dice_mean}, median={dice_median}, std={dice_std}, "
            f"tre mean={tre_mean}, median={tre_median}, std={tre_std}, \n"
        )


def build_config(config_path: (str, list), log_dir: str, ckpt_path: str) -> [dict, str]:
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
    log_dir = build_log_dir(log_dir)

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
    gpu,
    gpu_allow_growth,
    ckpt_path,
    mode,
    batch_size,
    log_dir,
    sample_label,
    config_path,
):
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
    :param config_path: to overwrite the default config
    """
    logging.error("TODO sample_label is not used in predict")

    # env vars
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false" if gpu_allow_growth else "true"

    # load config
    config, log_dir = build_config(
        config_path=config_path, log_dir=log_dir, ckpt_path=ckpt_path
    )
    preprocess_config = config["train"]["preprocess"]
    preprocess_config["batch_size"] = batch_size

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
        save_dir=log_dir + "/test",
    )

    # close the opened files in data loaders
    data_loader.close()


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
        args.gpu,
        args.gpu_allow_growth,
        args.ckpt_path,
        args.mode,
        args.batch_size,
        args.log_dir,
        args.sample_label,
        args.config_path,
    )


if __name__ == "__main__":
    main()
