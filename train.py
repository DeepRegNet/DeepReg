import argparse
import os
from datetime import datetime

import tensorflow as tf

import src.config.parser as config_parser
import src.data.loader as data_loader
import src.model.layer_util as layer_util
import src.model.network as network
import src.model.optimizer as opt
import src.model.step as step

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path of config", default="")
    args = parser.parse_args()

    # load config
    config = config_parser.load(args.config)
    data_config = config["data"]
    tf_data_config = config["tf"]["data"]
    tf_opt_config = config["tf"]["opt"]
    tf_model_config = config["tf"]["model"]
    tf_loss_config = config["tf"]["loss"]
    num_epochs = config["tf"]["epochs"]
    save_period = config["tf"]["save_period"]
    log_dir = config["log_dir"][:-1] if config["log_dir"][-1] == "/" else config["log_dir"]
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if config["tf"]["TF_FORCE_GPU_ALLOW_GROWTH"] else "false"

    # output
    log_dir = log_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tb_log_dir = log_dir + "/tensorboard"
    tb_writer_train = tf.summary.create_file_writer(tb_log_dir + "/train")
    tb_writer_test = tf.summary.create_file_writer(tb_log_dir + "/test")
    checkpoint_log_dir = log_dir + "/checkpoint"
    checkpoint_path = checkpoint_log_dir + "/cp-{epoch:d}.ckpt"

    # data
    data_loader_train, data_loader_test = data_loader.get_train_test_dataset(data_config)
    dataset_train = data_loader_train.get_dataset(training=True, **tf_data_config)
    dataset_test = data_loader_test.get_dataset(training=False, **tf_data_config)

    # optimizer
    optimizer = opt.get_optimizer(tf_opt_config)

    # model
    reg_model = network.build_model(moving_image_size=data_loader_train.moving_image_shape,
                                    fixed_image_size=data_loader_train.fixed_image_shape,
                                    batch_size=tf_data_config["batch_size"],
                                    tf_model_config=tf_model_config,
                                    tf_loss_config=tf_loss_config)

    # steps
    fixed_grid_ref = layer_util.get_reference_grid(grid_size=data_loader_train.fixed_image_shape)
    metrics_train, metrics_test = step.init_metrics()
    for epoch in range(num_epochs):
        print("%s | Start of epoch %d" % (datetime.now(), epoch))

        # train
        with tb_writer_train.as_default():
            for step_index, (inputs_train, labels_train, indices_train) in enumerate(dataset_train):
                metric_value_dict_train = step.train_step(model=reg_model, optimizer=optimizer,
                                                          inputs=inputs_train, labels=labels_train,
                                                          fixed_grid_ref=fixed_grid_ref,
                                                          tf_loss_config=tf_loss_config)

                # update metrics
                metrics_train.update(metric_value_dict=metric_value_dict_train)
                # update tensorboard
                metrics_train.update_tensorboard(step=optimizer.iterations)
            print("Training loss at epoch %d: %s" % (epoch, metrics_train))

        # test
        with tb_writer_test.as_default():
            for step_index, (inputs_test, labels_test, indices_test) in enumerate(dataset_test):
                metric_value_dict_test = step.eval_step(model=reg_model,
                                                        inputs=inputs_test, labels=labels_test,
                                                        fixed_grid_ref=fixed_grid_ref,
                                                        tf_loss_config=tf_loss_config)
                # update metrics
                metrics_test.update(metric_value_dict=metric_value_dict_test)
            # update tensorboard
            metrics_test.update_tensorboard(step=optimizer.iterations)
            print("Test loss at step %d: %s" % (step_index, metrics_test))

        # save models
        if epoch % save_period == 0:
            print("Save checkpoint at epoch %d" % epoch)
            reg_model.save_weights(filepath=checkpoint_path.format(epoch=epoch))

        # reset metrics
        metrics_train.reset()
        metrics_test.reset()
