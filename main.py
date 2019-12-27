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
num_epochs = 10

logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# data
data_loader = loader.PairedDataLoader(moving_image_dir, fixed_image_dir, moving_label_dir, fixed_label_dir)
dataset = data_loader.get_dataset()
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# model
local_model = network.LocalModel(batch_size=batch_size, num_channel_initial=num_channel_initial)
local_model.compile(optimizer, loss=loss.loss_fn)

# train
local_model.fit(dataset, epochs=num_epochs, callbacks=[tensorboard_callback])
