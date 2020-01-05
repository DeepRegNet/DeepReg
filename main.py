from datetime import datetime

import tensorflow as tf

import src.data.loader as loader
import src.model.loss as loss
import src.model.metric as metric
import src.model.network as network

# config
moving_image_dir = "./data/train/mr_images"
fixed_image_dir = "./data/train/us_images"
moving_label_dir = "./data/train/mr_labels"
fixed_label_dir = "./data/train/us_labels"

batch_size = 4
num_channel_initial = 16
learning_rate = 1.0e-5
num_epochs = 20000
dataset_shuffle_buffer_size = 1024
ddf_levels = [0, 1, 2, 3, 4]  # config in old code doesnt work

tb_log_dir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
train_summary_writer = tf.summary.create_file_writer(tb_log_dir)

# data
data_loader_train = loader.PairedDataLoader(moving_image_dir, fixed_image_dir, moving_label_dir, fixed_label_dir)
dataset_train = data_loader_train.get_dataset(batch_size=batch_size, training=True,
                                              dataset_shuffle_buffer_size=dataset_shuffle_buffer_size)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
# metrics
metrics = metric.Metrics(tb_names=dict(
    loss_sim="loss/similarity",
    loss_reg="loss/regularization",
    loss_total="loss/total",
    metric_dice="metric/dice",
    metric_dist="metric/centroid_distance",
    opt_lr="opt/learning_rate",
))

# model
local_model = network.build_model(moving_image_size=data_loader_train.moving_image_shape,
                                  fixed_image_size=data_loader_train.fixed_label_shape,
                                  batch_size=batch_size, num_channel_initial=num_channel_initial,
                                  ddf_levels=ddf_levels)
local_model.compile(optimizer, loss=loss.loss_similarity_fn)


# steps
@tf.function
def train_step(model, opt, inputs, labels):
    # forward
    with tf.GradientTape() as tape:
        predictions = model(inputs=inputs, training=True)

        # loss
        loss_sim_value = loss.loss_similarity_fn(y_true=labels, y_pred=predictions)
        loss_reg_value = sum(model.losses)
        loss_total_value = loss_sim_value + loss_reg_value

        # metrics
        metric_dice_value = loss.binary_dice(labels, predictions)
        metric_dist_value = loss.compute_centroid_distance(labels, predictions, data_loader_train.fixed_label_shape)

    # optimize
    grads = tape.gradient(loss_total_value, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))

    opt_lr_value = opt._decayed_lr('float32')

    return dict(
        loss_sim=loss_sim_value,
        loss_reg=loss_reg_value,
        loss_total=loss_total_value,
        metric_dice=metric_dice_value,
        metric_dist=metric_dist_value,
        opt_lr=opt_lr_value,
    )


# train
# print(local_model.summary())

with train_summary_writer.as_default():
    for epoch in range(num_epochs):
        print("Start of epoch %d" % (epoch,))

        for step, (inputs_train, labels_train) in enumerate(dataset_train):
            metric_value_dict = train_step(model=local_model, opt=optimizer,
                                           inputs=inputs_train, labels=labels_train)

            # update metrics
            metrics.update(metric_value_dict=metric_value_dict)
            # update tensorboard
            metrics.update_tensorboard(step=optimizer.iterations)

            print('Training loss at step %d: %s' % (step, metrics))

        print('Training loss at epoch %d: %s' % (epoch, metrics))

        # reset metrics
        metrics.reset()

# TODO organize graph in tensorboard
# TODO add test
# organize tensorboard
# track median/mean/min/max of dice and dist
# track median/mean/min/max of ddf, preds, labels
# note the sample id
# save model regularly
# add params
# log lr
