import logging
import os
from datetime import datetime

import click
import tensorflow as tf

import deepreg.config.parser as config_parser
import deepreg.model.loss.label as label_loss
import deepreg.model.metric as metric
import deepreg.model.optimizer as opt
from deepreg.data.load import get_data_loader
from deepreg.model.network.build import build_model


def init(config_path, log_dir, ckpt_path):
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
    # env vars
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
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
    dataset_train = data_loader_train.get_dataset_and_preprocess(training=True, repeat=True, **tf_data_config)
    dataset_val = data_loader_val.get_dataset_and_preprocess(training=False, repeat=True, **tf_data_config)
    dataset_size_train = data_loader_train.num_samples
    dataset_size_val = data_loader_val.num_samples

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
        model_outputs_names = ["tf_op_layer_output_ddf", "tf_op_layer_output_pred_fixed_label"]
        loss_fn = dict(zip(model_outputs_names,
                           [None,
                            label_loss.get_similarity_fn(config=tf_loss_config["similarity"]["label"]),
                            ]))

        metrics = dict(zip(model_outputs_names,
                           [None,
                            [
                                metric.MeanDiceScore(),
                                metric.MeanCentroidDistance(grid_size=data_loader_train.fixed_image_shape),
                                metric.MeanForegroundProportion(pred=False),
                                metric.MeanForegroundProportion(pred=True),
                            ],
                            ]))
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=metrics)
        print(model.summary())

        # load weights
        if ckpt_path != "":
            model.load_weights(ckpt_path)

        # train
        # callbacks
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=log_dir + "/save/weights-epoch{epoch:d}.ckpt", save_weights_only=True,
            period=save_period)
        # it's necessary to define the steps_per_epoch and validation_steps to prevent errors like
        # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
        model.fit(
            x=dataset_train,
            steps_per_epoch=dataset_size_train // tf_data_config["batch_size"],
            epochs=num_epochs,
            validation_data=dataset_val,
            validation_steps=dataset_size_val // tf_data_config["batch_size"],
            callbacks=[tensorboard_callback, checkpoint_callback],
        )


@click.command()
@click.option(
    "--gpu", "-g",
    help="GPU index",
    type=str,
    required=True,
)
@click.option(
    "--config_path", "-c",
    help="Path of config",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
)
@click.option(
    "--gpu_allow_growth/--not_gpu_allow_growth",
    help="Do not take all GPU memory",
    default=False,
    show_default=True)
@click.option(
    "--ckpt_path",
    help="Path of checkpoint to load",
    default="",
    show_default=True,
    type=str,
)
@click.option(
    "--log_dir",
    help="Path of log directory",
    default="",
    show_default=True,
    type=str,
)
def main(gpu, config_path, gpu_allow_growth, ckpt_path, log_dir):
    train(gpu, config_path, gpu_allow_growth, ckpt_path, log_dir)


if __name__ == "__main__":
    main()
