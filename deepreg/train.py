# coding=utf-8

"""
Module to train a network using init files and a CLI.
"""

import argparse
import os

import tensorflow as tf

import deepreg.model.optimizer as opt
import deepreg.parser as config_parser
from deepreg.model.network.build import build_model
from deepreg.util import build_dataset, build_log_dir


def build_config(
    config_path: (str, list),
    log_root: str,
    log_dir: str,
    ckpt_path: str,
    max_epochs: int = -1,
) -> [dict, str]:
    """
    Function to initialise log directories,
    assert that checkpointed model is the right
    type and to parse the configuration for training.

    :param config_path: list of str, path to config file
    :param log_root: str, root of logs
    :param log_dir: str, path to where training logs to be stored.
    :param ckpt_path: str, path where model is stored.
    :param max_epochs: int, if max_epochs > 0, will use it to overwrite the configuration
    :return: - config: a dictionary saving configuration
             - log_dir: the path of directory to save logs
    """

    # init log directory
    log_dir = build_log_dir(log_root=log_root, log_dir=log_dir)

    # check checkpoint path
    if ckpt_path != "":
        if not ckpt_path.endswith(".ckpt"):
            raise ValueError(f"checkpoint path should end with .ckpt, got {ckpt_path}")

    # load config
    config = config_parser.load_configs(config_path)

    # overwrite epochs and save_period if necessary
    if max_epochs > 0:
        config["train"]["epochs"] = max_epochs
        config["train"]["save_period"] = min(max_epochs, config["train"]["save_period"])

    # backup config
    config_parser.save(config=config, out_dir=log_dir)

    # batch_size in original config corresponds to batch_size per GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    config["train"]["preprocess"]["batch_size"] *= max(len(gpus), 1)

    return config, log_dir


def build_callbacks(log_dir: str, histogram_freq: int, save_period: int) -> list:
    """
    Function to prepare callbacks for training.

    :param log_dir: directory of logs
    :param histogram_freq: save the histogram every X epochs
    :param save_period: save the checkpoint every X epochs
    :return: a list of callbacks
    """
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=histogram_freq
    )
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir + "/save/weights-epoch{epoch:d}.ckpt",
        save_weights_only=True,
        period=save_period,
    )
    return [tensorboard_callback, checkpoint_callback]


def train(
    gpu: str,
    config_path: (str, list),
    gpu_allow_growth: bool,
    ckpt_path: str,
    log_dir: str,
    log_root: str = "logs",
    max_epochs: int = -1,
):
    """
    Function to train a model.

    :param gpu: str, which local gpu to use to train
    :param config_path: str, path to configuration set up
    :param gpu_allow_growth: bool, whether or not to allocate whole GPU memory to training
    :param ckpt_path: str, where to store training checkpoints
    :param log_root: str, root of logs
    :param log_dir: str, where to store logs in training
    :param max_epochs: int, if max_epochs > 0, will use it to overwrite the configuration
    """
    # set env variables
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if gpu_allow_growth else "false"

    # load config
    config, log_dir = build_config(
        config_path=config_path,
        log_root=log_root,
        log_dir=log_dir,
        ckpt_path=ckpt_path,
        max_epochs=max_epochs,
    )

    # build dataset
    data_loader_train, dataset_train, steps_per_epoch_train = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        mode="train",
        training=True,
        repeat=True,
    )
    assert data_loader_train is not None  # train data should not be None
    data_loader_val, dataset_val, steps_per_epoch_val = build_dataset(
        dataset_config=config["dataset"],
        preprocess_config=config["train"]["preprocess"],
        mode="valid",
        training=False,
        repeat=True,
    )

    # build callbacks
    callbacks = build_callbacks(
        log_dir=log_dir,
        histogram_freq=config["train"][
            "save_period"
        ],  # use save_period for histogram_freq
        save_period=config["train"]["save_period"],
    )

    # use strategy to support multiple GPUs
    # the network is mirrored in each GPU so that we can use larger batch size
    # https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_tfkerasmodelfit
    # only model, optimizer and metrics need to be defined inside the strategy
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = build_model(
            moving_image_size=data_loader_train.moving_image_shape,
            fixed_image_size=data_loader_train.fixed_image_shape,
            index_size=data_loader_train.num_indices,
            labeled=config["dataset"]["labeled"],
            batch_size=config["train"]["preprocess"]["batch_size"],
            model_config=config["train"]["model"],
            loss_config=config["train"]["loss"],
        )
        optimizer = opt.build_optimizer(optimizer_config=config["train"]["optimizer"])

    # compile
    model.compile(optimizer=optimizer)

    # load weights
    if ckpt_path != "":
        model.load_weights(ckpt_path)

    # train
    # it's necessary to define the steps_per_epoch and validation_steps to prevent errors like
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    model.fit(
        x=dataset_train,
        steps_per_epoch=steps_per_epoch_train,
        epochs=config["train"]["epochs"],
        validation_data=dataset_val,
        validation_steps=steps_per_epoch_val,
        callbacks=callbacks,
    )

    # close file loaders in data loaders after training
    data_loader_train.close()
    if data_loader_val is not None:
        data_loader_val.close()


def main(args=None):
    """Entry point for train script."""

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
        help="Path of the saved model checkpoint to load."
        "No need to provide if start training from scratch.",
        default="",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--log_root", help="Root of log directory.", default="logs", type=str
    )

    parser.add_argument(
        "--log_dir",
        "-l",
        help="Name of log directory. The directory is under log root, e.g. logs/ by default."
        "If not provided, a timestamp based folder will be created.",
        default="",
        type=str,
    )

    parser.add_argument(
        "--config_path",
        "-c",
        help="Path of config, must end with .yaml. Can pass multiple paths.",
        type=str,
        nargs="+",
        required=True,
    )

    parser.add_argument(
        "--max_epochs",
        help="The maximum number of epochs, -1 means following configuration.",
        type=int,
        default=-1,
    )

    args = parser.parse_args(args)
    train(
        gpu=args.gpu,
        config_path=args.config_path,
        gpu_allow_growth=args.gpu_allow_growth,
        ckpt_path=args.ckpt_path,
        log_root=args.log_root,
        log_dir=args.log_dir,
        max_epochs=args.max_epochs,
    )


if __name__ == "__main__":
    main()
