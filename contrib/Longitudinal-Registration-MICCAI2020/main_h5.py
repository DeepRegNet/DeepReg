import os
from datetime import datetime

import numpy as np
import src.data.loader_h5 as loader
import src.model.layer_util as layer_util
import src.model.loss as loss
import src.model.metric as metric
import src.model.network as network
import steps as steps
import tensorflow as tf
import utils as utils
from config import args

# gpu config
GPUs = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(GPUs[args.gpu], "GPU")
if GPUs and args.gpu_memory_control:
    for gpu in GPUs:
        tf.config.experimental.set_memory_growth(gpu, True)

# prepare data
data_loader_train = loader.H5DataLoader(args, phase="train")
dataset_train = data_loader_train.get_dataset(batch_size=args.batch_size)
data_loader_test = loader.H5DataLoader(args, phase="test")
dataset_test = data_loader_test.get_dataset(batch_size=args.batch_size)

# set tensorboard
if args.continue_epoch == "-1":
    start_epoch = 0
    log_dir = os.path.join(
        args.log_dir, args.exp_name + "-" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )
else:
    log_dir, breakpoint_ckpt_path, start_epoch = utils.get_exp_dir_and_ckpt(args)
tb_log_dir = os.path.join(log_dir, "tensorboard")
os.makedirs(tb_log_dir, exist_ok=True)
tb_writer_train = tf.summary.create_file_writer(tb_log_dir + "/train")
with tb_writer_train.as_default():
    tf.summary.text(name="Experiment Configures", data=utils.format_args(args), step=0)
tb_writer_test = tf.summary.create_file_writer(tb_log_dir + "/test")
with tb_writer_test.as_default():
    tf.summary.text(name="Experiment Configures", data=utils.format_args(args), step=0)
checkpoint_log_dir = os.path.join(log_dir, "checkpoint")
checkpoint_path = checkpoint_log_dir + "/cp-{epoch:04d}.ckpt"

# set model
local_model = network.build_model(
    moving_image_size=data_loader_train.moving_image_shape,
    fixed_image_size=data_loader_train.fixed_image_shape,
    batch_size=args.batch_size,
    num_channel_initial=args.num_channel_initial,
    ddf_levels=args.ddf_levels,
    ddf_energy_type=args.ddf_energy_type,
)  # fix it, use args
if args.continue_epoch != "-1":
    local_model.load_weights(breakpoint_ckpt_path)
    print(f"loading weights from {breakpoint_ckpt_path}")

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

# metrics
tb_names_test = dict(
    loss_sim="loss/similarity",
    loss_reg="loss/regularization",
    loss_mmd="loss/mmd",
    loss_dice="loss/dice",
    loss_total="loss/total",
    metric_dice="metric/dice",
    metric_dist="metric/centroid_distance",
)
tb_names_train = dict(**tb_names_test, opt_lr="opt/learning_rate")
metrics_train = metric.Metrics(tb_names=tb_names_train)
metrics_test = metric.Metrics(tb_names=tb_names_test)


if args.loss_type == "ncc":
    local_model.compile(optimizer, loss=loss.loss_ncc)
elif args.loss_type == "ssd":
    local_model.compile(optimizer, loss=loss.loss_ssd)
elif args.loss_type == "gmi":
    local_model.compile(optimizer, loss=loss.loss_global_mutual_information)
else:
    print("loss type wrong")
    raise NotImplementedError

# tf.keras.utils.plot_model(local_model, to_file='model.png', show_shapes=True, show_layer_names=True,
#                           rankdir='TB',  # tb means vertical, LR means horizontal
#                           dpi=300)

# steps
# print(local_model.summary())
fixed_grid_ref = layer_util.get_reference_grid(
    grid_size=data_loader_train.fixed_image_shape
)

for epoch in range(start_epoch, args.epochs):
    print("Start of epoch %d" % (epoch,))
    # [print(i) for i in data_loader_train.key_pairs_list]
    # train
    with tb_writer_train.as_default():
        for step, (inputs, fixed_label, indices) in enumerate(dataset_train):
            # print(inputs[0].shape)
            # break
            metric_value_dict_train = steps.train_step(
                args_dict=args.__dict__,
                model=local_model,
                optimizer=optimizer,
                inputs=inputs,
                labels=fixed_label,
                fixed_grid_ref=fixed_grid_ref,
            )
            # update metrics
            metrics_train.update(metric_value_dict=metric_value_dict_train)
            # update tensorboard
            metrics_train.update_tensorboard(step=optimizer.iterations)
            # input_imgs = np.transpose(inputs_train[0][0], (2, 0, 1))
            # input_imgs = np.expand_dims(input_imgs, axis=3)
            # tf.summary.image("training data", input_imgs, max_outputs=25, step=optimizer.iterations)
            print("Training loss at step %d: %s" % (step, metrics_train))
        print("Training loss at epoch %d: %s" % (epoch, metrics_train))

    # print('----------epoch ends--------------')
    # continue

    # test
    with tb_writer_test.as_default():
        for step, (inputs, fixed_label, indices_test) in enumerate(dataset_test):
            metric_value_dict_test = steps.valid_step(
                args_dict=args.__dict__,
                model=local_model,
                inputs=inputs,
                labels=fixed_label,
                fixed_grid_ref=fixed_grid_ref,
            )
            # update metrics
            metrics_test.update(metric_value_dict=metric_value_dict_test)
        # update tensorboard
        metrics_test.update_tensorboard(step=optimizer.iterations)
        print("Test loss at step %d: %s" % (step, metrics_test))

    # save models
    if (epoch + 1) % args.save_period == 0:
        print("Save checkpoint at epoch %d" % epoch)
        local_model.save_weights(filepath=checkpoint_path.format(epoch=epoch))

    # should send data back...
    # reset metrics
    metrics_train.reset()
    metrics_test.reset()
