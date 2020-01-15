import os
from datetime import datetime

import tensorflow as tf

import src.model.layer_util as layer_util
import src.model.loss as loss
import src.model.metric as metric
import src.model.network as network
import steps as steps
from src.data.loader_h5 import H5DataLoader
from src.data.loader_nifti import NiftiDataLoader


def get_train_test_dataset(option, load_into_memory):
    if option == "demo":
        moving_image_dir_train = "./data/demo/train/mr_images"
        fixed_image_dir_train = "./data/demo/train/us_images"
        moving_label_dir_train = "./data/demo/train/mr_labels"
        fixed_label_dir_train = "./data/demo/train/us_labels"

        moving_image_dir_test = "./data/demo/test/mr_images"
        fixed_image_dir_test = "./data/demo/test/us_images"
        moving_label_dir_test = "./data/demo/test/mr_labels"
        fixed_label_dir_test = "./data/demo/test/us_labels"

        data_loader_train = NiftiDataLoader(moving_image_dir_train, fixed_image_dir_train,
                                            moving_label_dir_train, fixed_label_dir_train,
                                            load_into_memory)

        data_loader_test = NiftiDataLoader(moving_image_dir_test, fixed_image_dir_test,
                                           moving_label_dir_test, fixed_label_dir_test,
                                           load_into_memory)
    elif option == "full_nifti":
        moving_image_dir_train = "./data/full/train/mr_images"
        fixed_image_dir_train = "./data/full/train/us_images"
        moving_label_dir_train = "./data/full/train/mr_labels"
        fixed_label_dir_train = "./data/full/train/us_labels"

        moving_image_dir_test = "./data/full/test/mr_images"
        fixed_image_dir_test = "./data/full/test/us_images"
        moving_label_dir_test = "./data/full/test/mr_labels"
        fixed_label_dir_test = "./data/full/test/us_labels"

        data_loader_train = NiftiDataLoader(moving_image_dir_train, fixed_image_dir_train,
                                            moving_label_dir_train, fixed_label_dir_train,
                                            load_into_memory)

        data_loader_test = NiftiDataLoader(moving_image_dir_test, fixed_image_dir_test,
                                           moving_label_dir_test, fixed_label_dir_test,
                                           load_into_memory)
    elif option == "full_h5":
        # TODO separate train/test
        moving_image_filename = "data/full_h5/mr_images_resampled800.h5"
        moving_label_filename = "data/full_h5/mr_labels_resampled800_post3.h5"
        fixed_image_filename = "data/full_h5/us_images_resampled800.h5"
        fixed_label_filename = "data/full_h5/us_labels_resampled800_post3.h5"
        seed = 0
        shuffle = True
        train_start_index = 0
        train_end_index = 100
        test_start_index = train_end_index
        test_end_index = 108
        data_loader_train = H5DataLoader(moving_image_filename=moving_image_filename,
                                         fixed_image_filename=fixed_image_filename,
                                         moving_label_filename=moving_label_filename,
                                         fixed_label_filename=fixed_label_filename,
                                         seed=seed,
                                         shuffle=shuffle,
                                         index_start=train_start_index,
                                         index_end=train_end_index)

        data_loader_test = H5DataLoader(moving_image_filename=moving_image_filename,
                                        fixed_image_filename=fixed_image_filename,
                                        moving_label_filename=moving_label_filename,
                                        fixed_label_filename=fixed_label_filename,
                                        seed=seed,
                                        shuffle=shuffle,
                                        index_start=test_start_index,
                                        index_end=test_end_index)
    else:
        raise ValueError("Unknown option")
    return data_loader_train, data_loader_test


# config
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
data_option = "full_h5"
data_load_into_memory = False
batch_size = 2
num_channel_initial = 4
learning_rate = 1.0e-5
num_epochs = 20
save_period = 5
dataset_shuffle_buffer_size = 1024
ddf_levels = [0, 1, 2, 3, 4]  # numbers should be <= 4
loss_type = "dice"  # not used
loss_scales = [0, 1, 2, 4, 8, 16, 32]  # not used

# output
log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tb_log_dir = log_dir + "/tensorboard"
tb_writer_train = tf.summary.create_file_writer(tb_log_dir + "/train")
tb_writer_test = tf.summary.create_file_writer(tb_log_dir + "/test")
checkpoint_log_dir = log_dir + "/checkpoint"
checkpoint_path = checkpoint_log_dir + "/cp-{epoch:04d}.ckpt"

# data
data_loader_train, data_loader_test = get_train_test_dataset(option=data_option, load_into_memory=data_load_into_memory)
dataset_train = data_loader_train.get_dataset(batch_size=batch_size, training=True,
                                              dataset_shuffle_buffer_size=dataset_shuffle_buffer_size)
dataset_test = data_loader_test.get_dataset(batch_size=batch_size, training=False,
                                            dataset_shuffle_buffer_size=dataset_shuffle_buffer_size)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# metrics
tb_names_test = dict(
    loss_sim="loss/similarity",
    loss_reg="loss/regularization",
    loss_total="loss/total",
    metric_dice="metric/dice",
    metric_dist="metric/centroid_distance",
)
tb_names_train = dict(
    **tb_names_test,
    opt_lr="opt/learning_rate",
)
metrics_train = metric.Metrics(tb_names=tb_names_train)
metrics_test = metric.Metrics(tb_names=tb_names_test)

# model
local_model = network.build_model(moving_image_size=data_loader_train.moving_image_shape,
                                  fixed_image_size=data_loader_train.fixed_image_shape,
                                  batch_size=batch_size, num_channel_initial=num_channel_initial,
                                  ddf_levels=ddf_levels)
local_model.compile(optimizer, loss=loss.loss_similarity_fn)

# steps
print(local_model.summary())

fixed_grid_ref = layer_util.get_reference_grid(grid_size=data_loader_train.fixed_image_shape)

for epoch in range(num_epochs):
    print("%s | Start of epoch %d" % (datetime.now(), epoch))

    # train
    with tb_writer_train.as_default():
        for step, (inputs_train, labels_train, indices_train) in enumerate(dataset_train):
            metric_value_dict_train = steps.train_step(model=local_model, optimizer=optimizer,
                                                       inputs=inputs_train, labels=labels_train,
                                                       fixed_grid_ref=fixed_grid_ref)

            # update metrics
            metrics_train.update(metric_value_dict=metric_value_dict_train)
            # update tensorboard
            metrics_train.update_tensorboard(step=optimizer.iterations)
        print("Training loss at epoch %d: %s" % (epoch, metrics_train))

    # test
    with tb_writer_test.as_default():
        for step, (inputs_test, labels_test, indices_test) in enumerate(dataset_test):
            metric_value_dict_test = steps.valid_step(model=local_model,
                                                      inputs=inputs_test, labels=labels_test,
                                                      fixed_grid_ref=fixed_grid_ref)
            # update metrics
            metrics_test.update(metric_value_dict=metric_value_dict_test)
        # update tensorboard
        metrics_test.update_tensorboard(step=optimizer.iterations)
        print("Test loss at step %d: %s" % (step, metrics_test))

    # save models
    if epoch % save_period == 0:
        print("Save checkpoint at epoch %d" % epoch)
        local_model.save_weights(filepath=checkpoint_path.format(epoch=epoch))

    # reset metrics
    metrics_train.reset()
    metrics_test.reset()

# TODO
# organize graph in tensorboard
# add params
# https://www.tensorflow.org/guide/distributed_training#using_tfdistributestrategy_with_keras
