"""
Module to train a network using init files and a CLI
"""

import logging
import os
from datetime import datetime

import argparse
import tensorflow as tf

import deepreg.config.parser as config_parser
import deepreg.model.optimizer as opt
from deepreg.data.load import get_data_loader
from deepreg.model.network.build import build_model


def init(config_path, log_dir, ckpt_path):
    """
    Function to initialise log directories,
    assert that checkpointed model is the right
    type and to parse the configuration for training
    :param config_path: str, path to config file
    :param log_dir: str, path to where training logs
                    to be stored.
    :param ckpt_path: str, path where model is stored.
    """
    # init log directory
    if log_dir == "":  # default
        log_dir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    if os.path.exists(log_dir):
        logging.warning("Log directory {} exists already.".format(log_dir))
    else:
        os.makedirs(log_dir)

    # check checkpoint path
    if ckpt_path != "":
        if not ckpt_path.endswith(".ckpt"):
            raise ValueError("checkpoint path should end with .ckpt")

    # load and backup config
    config = config_parser.load(config_path)

    config_parser.save(config=config, out_dir=log_dir)
    return config, log_dir


def train(gpu, config_path, gpu_allow_growth, ckpt_path, log_dir):
    """
    Function to train a model
    :param gpu: str, which local gpu to use to train
    :param config_path: str, path to configuration set up
    :param gpu_allow_growth: bool, whether or not to allocate
                             whole GPU memory to training
    :param ckpt_path: str, where to store training ckpts
    :param log_dir: str, where to store logs in training
    """

    # env vars
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu if gpu != '-1' else ''
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if gpu_allow_growth else "false"

    # load config
    config, log_dir = init(config_path, log_dir, ckpt_path)
    data_config = config["data"]
    tf_data_config = config["tf"]["data"]
    tf_opt_config = config["tf"]["opt"]
    tf_model_config = config["tf"]["model"]
    tf_loss_config = config["tf"]["loss"]
    num_epochs = config["tf"]["epochs"]
    save_period = config["tf"]["save_period"]
    histogram_freq = config["tf"]["histogram_freq"]

    # data
    data_loader_train = get_data_loader(data_config, "train")
    data_loader_val = get_data_loader(data_config, "valid")
    dataset_train = data_loader_train.get_dataset_and_preprocess(training=True,
                                                                 repeat=True,
                                                                 **tf_data_config)
    dataset_val = data_loader_val.get_dataset_and_preprocess(training=False,
                                                             repeat=True,
                                                             **tf_data_config)
    dataset_size_train = data_loader_train.num_samples
    dataset_size_val = data_loader_val.num_samples
    steps_per_epoch_train = max(dataset_size_train // tf_data_config["batch_size"], 1)
    steps_per_epoch_valid = max(dataset_size_val // tf_data_config["batch_size"], 1)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # model
        model = build_model(moving_image_size=data_loader_train.moving_image_shape,
                            fixed_image_size=data_loader_train.fixed_image_shape,
                            index_size=data_loader_train.num_indices,
                            labeled=data_config["labeled"],
                            batch_size=tf_data_config["batch_size"],
                            tf_model_config=tf_model_config,
                            tf_loss_config=tf_loss_config)
        model.summary()

        # compile
        optimizer = opt.get_optimizer(tf_opt_config)

        model.compile(optimizer=optimizer)
        print(model.summary())

        # load weights
        if ckpt_path is None:
            model.load_weights(ckpt_path)

        # train
        # callbacks
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                              histogram_freq=histogram_freq)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + "/save/weights-epoch{epoch:d}.ckpt", save_weights_only=True,
            period=save_period)
        # it's necessary to define the steps_per_epoch and validation_steps to prevent errors like
        # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
        model.fit(
            x=dataset_train,
            steps_per_epoch=steps_per_epoch_train,
            epochs=num_epochs,
            validation_data=dataset_val,
            validation_steps=steps_per_epoch_valid,
            callbacks=[tensorboard_callback, checkpoint_callback],
        )


def main(args=None):
    """Entry point for train script"""

    parser = argparse.ArgumentParser(description="train",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## ADD POSITIONAL ARGUMENTS
    parser.add_argument("--config_path",
                        "-c",
                        help="Path of config",
                        type=str,
                        required=True)


    parser.add_argument("--gpu",
                        "-g",
                        help="GPU index for training, multiple gpus can be passed",
                        default='-1',
                        type=str,
                        # nargs='+',
                        required=False)

    parser.add_argument("--gpu_allow_growth",
                        "-gr",
                        help="Prevent TensorFlow from reserving all available GPU memory",
                        default=False)

    parser.add_argument("--ckpt_path",
                        "-k",
                        help="Path of checkpointed model to load",
                        default="",
                        type=str,
                        required=False)

    parser.add_argument("--log_dir",
                        "-l",
                        help="Path of log directory",
                        default="",
                        type=str,
                        required=False)


    args = parser.parse_args(args)
    train(args.gpu,
          args.config_path,
          args.gpu_allow_growth,
          args.ckpt_path,
          args.log_dir)


if __name__ == "__main__":
    main()
