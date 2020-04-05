import argparse
import os
from datetime import datetime

import tensorflow as tf

import deepreg.config.parser as config_parser
import deepreg.data.mr.load as data_loader
import deepreg.model.loss.label as label_loss
import deepreg.model.metric as metric
import deepreg.model.network as network
import deepreg.model.optimizer as opt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", help="GPU index", required=True)
    parser.add_argument("-c", "--config", help="Path of config", required=True)
    parser.add_argument("-m", "--memory", dest="memory", action='store_true', help="do not take all GPU memory")
    parser.add_argument("--ckpt", help="Path of checkpoint to load", default="")
    parser.add_argument("-l", "--log", help="Name of log folder", default="")
    parser.set_defaults(memory=False)
    args = parser.parse_args()

    # env vars
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if args.memory else "false"

    # load config
    config = config_parser.load(args.config)
    data_config = config["data"]
    tf_data_config = config["tf"]["data"]
    tf_opt_config = config["tf"]["opt"]
    tf_model_config = config["tf"]["model"]
    tf_loss_config = config["tf"]["loss"]
    num_epochs = config["tf"]["epochs"]
    save_period = config["tf"]["save_period"]
    histogram_freq = config["tf"]["histogram_freq"]
    log_dir = config["log_dir"][:-1] if config["log_dir"][-1] == "/" else config["log_dir"]

    # output
    log_folder_name = args.log if args.log != "" else datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = log_dir + "/" + log_folder_name

    checkpoint_init_path = args.ckpt
    if checkpoint_init_path != "":
        if not checkpoint_init_path.endswith(".ckpt"):
            raise ValueError("checkpoint path should end with .ckpt")

    # backup config
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    config_parser.save(config=config, out_dir=log_dir)

    # data
    data_loader_train, data_loader_val, _ = data_loader.get_data_loaders(
        data_type="segmentation", data_config=data_config)
    dataset_train = data_loader_train.get_dataset(training=True, repeat=True, **tf_data_config)
    dataset_val = data_loader_val.get_dataset(training=False, repeat=True, **tf_data_config)
    dataset_size_train = data_loader_train.num_images
    dataset_size_val = data_loader_val.num_images

    # optimizer
    optimizer = opt.get_optimizer(tf_opt_config)

    # callbacks
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=histogram_freq)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=log_dir + "/save/weights-epoch{epoch:d}.ckpt", save_weights_only=True,
        period=save_period)

    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # model
        reg_model = network.build_model(moving_image_size=data_loader_train.moving_image_shape,
                                        fixed_image_size=data_loader_train.fixed_image_shape,
                                        index_size=data_loader_train.num_indices,
                                        batch_size=tf_data_config["batch_size"],
                                        tf_model_config=tf_model_config,
                                        tf_loss_config=tf_loss_config)

        # metrics
        reg_model.compile(optimizer=optimizer,
                          loss=label_loss.get_similarity_fn(config=tf_loss_config["similarity"]["label"]),
                          metrics=[metric.MeanDiceScore(),
                                   metric.MeanCentroidDistance(grid_size=data_loader_train.fixed_image_shape),
                                   metric.MeanForegroundProportion(pred=False),
                                   metric.MeanForegroundProportion(pred=True),
                                   ])
        print(reg_model.summary())

        # load weights
        if checkpoint_init_path != "":
            reg_model.load_weights(checkpoint_init_path)

        # train
        # it's necessary to define the steps_per_epoch and validation_steps to prevent errors like
        # BaseCollectiveExecutor::StartAbort Out of range: End of sequence
        reg_model.fit(
            x=dataset_train,
            steps_per_epoch=dataset_size_train // tf_data_config["batch_size"],
            epochs=num_epochs,
            validation_data=dataset_val,
            validation_steps=dataset_size_val // tf_data_config["batch_size"],
            callbacks=[tensorboard_callback, checkpoint_callback],
        )
