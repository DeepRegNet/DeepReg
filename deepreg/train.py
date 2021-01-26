# coding=utf-8

"""
Module to train a network using init files and a CLI.
"""

import argparse
import os

import tensorflow as tf

import deepreg.config.parser as config_parser
import deepreg.model.optimizer as opt
from deepreg.callback import build_checkpoint_callback
from deepreg.registry import REGISTRY
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
    :param log_root: root of logs
    :param log_dir: path to where training logs to be stored.
    :param ckpt_path: path where model is stored.
    :param max_epochs: if max_epochs > 0, use it to overwrite the configuration
    :return: - config: a dictionary saving configuration
             - log_dir: the path of directory to save logs
    """

    # init log directory
    log_dir = build_log_dir(log_root=log_root, log_dir=log_dir)

    # load config
    config = config_parser.load_configs(config_path)

    # replace the ~ with user home path
    ckpt_path = os.path.expanduser(ckpt_path)

    # overwrite epochs and save_period if necessary
    if max_epochs > 0:
        config["train"]["epochs"] = max_epochs
        config["train"]["save_period"] = min(max_epochs, config["train"]["save_period"])

    # backup config
    config_parser.save(config=config, out_dir=log_dir)

    # batch_size in original config corresponds to batch_size per GPU
    gpus = tf.config.experimental.list_physical_devices("GPU")
    config["train"]["preprocess"]["batch_size"] *= max(len(gpus), 1)

    return config, log_dir, ckpt_path


def train(
    gpu: str,
    config_path: (str, list),
    gpu_allow_growth: bool,
    ckpt_path: str,
    log_dir: str = "",
    log_root: str = "logs",
    max_epochs: int = -1,
):
    """
    Function to train a model.

    :param gpu: which local gpu to use to train
    :param config_path: path to configuration set up
    :param gpu_allow_growth: whether to allocate whole GPU memory for training
    :param ckpt_path: where to store training checkpoints
    :param log_root: root of logs
    :param log_dir: where to store logs in training
    :param max_epochs: if max_epochs > 0, will use it to overwrite the configuration
    """
    # set env variables
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if gpu_allow_growth else "false"

    # load config
    config, log_dir, ckpt_path = build_config(
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

    # use strategy to support multiple GPUs
    # the network is mirrored in each GPU so that we can use larger batch size
    # https://www.tensorflow.org/guide/distributed_training
    # only model, optimizer and metrics need to be defined inside the strategy
    num_devices = max(len(tf.config.list_physical_devices("GPU")), 1)
    if num_devices > 1:
        strategy = tf.distribute.MirroredStrategy()  # pragma: no cover
    else:
        strategy = tf.distribute.get_strategy()
    with strategy.scope():
        model = REGISTRY.build_model(
            config=dict(
                name=config["train"]["method"],
                moving_image_size=data_loader_train.moving_image_shape,
                fixed_image_size=data_loader_train.fixed_image_shape,
                index_size=data_loader_train.num_indices,
                labeled=config["dataset"]["labeled"],
                batch_size=config["train"]["preprocess"]["batch_size"],
                config=config["train"],
                num_devices=num_devices,
            )
        )
        optimizer = opt.build_optimizer(optimizer_config=config["train"]["optimizer"])

    # compile
    model.compile(optimizer=optimizer)

    # build callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, histogram_freq=config["train"]["save_period"]
    )
    ckpt_callback, initial_epoch = build_checkpoint_callback(
        model=model,
        dataset=dataset_train,
        log_dir=log_dir,
        save_period=config["train"]["save_period"],
        ckpt_path=ckpt_path,
    )
    callbacks = [tensorboard_callback, ckpt_callback]

    # train
    # it's necessary to define the steps_per_epoch
    # and validation_steps to prevent errors like
    # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
    model.fit(
        x=dataset_train,
        steps_per_epoch=steps_per_epoch_train,
        initial_epoch=initial_epoch,
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
    """
    Entry point for train script.

    :param args:
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
        help="Name of log directory."
        "The directory is under log root, e.g. logs/ by default."
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
    main()  # pragma: no cover
