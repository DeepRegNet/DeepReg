from datetime import datetime

import tensorflow as tf

import src.data.loader as loader
import src.model.loss as loss
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

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# data
data_loader = loader.PairedDataLoader(moving_image_dir, fixed_image_dir, moving_label_dir, fixed_label_dir)
dataset = data_loader.get_dataset()
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# model
local_model = network.build_model(moving_image_size=data_loader.moving_image_shape,
                                  fixed_image_size=data_loader.fixed_label_shape,
                                  batch_size=batch_size, num_channel_initial=num_channel_initial)
local_model.compile(optimizer, loss=loss.multi_scale_loss_fn)

# train
# local_model.fit(dataset, epochs=num_epochs, callbacks=[tensorboard_callback])
print(data_loader.moving_image_shape, data_loader.fixed_label_shape)
local_model.build(
    input_shape=[[batch_size] + data_loader.moving_image_shape, [batch_size] + data_loader.fixed_label_shape,
                 [batch_size] + data_loader.moving_image_shape])
print(local_model.summary())

for step, (x_batch_train, y_batch_train) in enumerate(dataset):
    with tf.GradientTape() as tape:
        out = local_model(x_batch_train)
        loss_value = loss.multi_scale_loss_fn(y_batch_train, out)
        loss_value += sum(local_model.losses)

    grads = tape.gradient(loss_value, local_model.trainable_weights)
    optimizer.apply_gradients(zip(grads, local_model.trainable_weights))
    print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
    print('Seen so far: %s samples' % ((step + 1) * 64))

# TODO add metrics
# TODO organize graph in tensorboard
