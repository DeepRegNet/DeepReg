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

batch_size = 2
num_channel_initial = 8
learning_rate = 1e-4
num_epochs = 2
dataset_shuffle_buffer_size = 1024
epochs = 3

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# data
data_loader_train = loader.PairedDataLoader(moving_image_dir, fixed_image_dir, moving_label_dir, fixed_label_dir)
dataset_train = data_loader_train.get_dataset(batch_size=batch_size, training=True,
                                              dataset_shuffle_buffer_size=dataset_shuffle_buffer_size)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# metrics
metrics = metric.Metrics(metric_names=[
    "loss_sim",
    "loss_reg",
    "loss_total",
])

# model
local_model = network.build_model(moving_image_size=data_loader_train.moving_image_shape,
                                  fixed_image_size=data_loader_train.fixed_label_shape,
                                  batch_size=batch_size, num_channel_initial=num_channel_initial)
local_model.compile(optimizer, loss=loss.loss_similarity_fn)


# steps
@tf.function
def train_step(model, inputs, labels):
    # forward
    with tf.GradientTape() as tape:
        predictions = model(inputs=inputs, training=True)
        loss_sim_value = loss.loss_similarity_fn(y_true=labels, y_pred=predictions)
        loss_reg_value = sum(model.losses)
        loss_total_value = loss_sim_value + loss_reg_value

    # optimize
    grads = tape.gradient(loss_total_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return dict(
        loss_sim=loss_sim_value,
        loss_reg=loss_reg_value,
        loss_total=loss_total_value
    )


# train
# local_model.fit(dataset, epochs=num_epochs, callbacks=[tensorboard_callback])

# print(local_model.summary())

for epoch in range(epochs):
    print("Start of epoch %d" % (epoch,))

    for step, (inputs_train, labels_train) in enumerate(dataset_train):
        metric_value_dict = train_step(model=local_model, inputs=inputs_train, labels=labels_train)

        # update metrics
        metrics.update(metric_value_dict=metric_value_dict)

        print('Training loss (for one batch) at step %s: %s' % (step, metrics))

    # reset metrics
    metrics.reset()

# TODO add metrics
# TODO organize graph in tensorboard
# TODO add DA
