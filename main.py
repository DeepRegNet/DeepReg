import tensorflow as tf

import src.data.loader as loader
import src.model.loss as loss
import src.model.network as network


def loss_fn(y_true, y_pred):
    """

    :param y_true: fixed_label
    :param y_pred: warped_moving_label
    :return:
    """
    fixed_label, warped_moving_label = y_true, y_pred
    if len(fixed_label.shape) == 4:
        fixed_label = tf.expand_dims(fixed_label, axis=4)
    if len(warped_moving_label.shape) == 4:
        warped_moving_label = tf.expand_dims(warped_moving_label, axis=4)
    loss_similarity = tf.reduce_mean(loss.multi_scale_loss(label_fixed=fixed_label,
                                                           label_moving=warped_moving_label,
                                                           loss_type="dice",
                                                           loss_scales=[0, 1, 2, 4, 8]))
    # loss_regularizer = tf.reduce_mean(loss.local_displacement_energy(ddf, "bending", 0.5))
    # total_loss = loss_similarity + loss_regularizer  # TODO add coeff for loss regularizer
    # return total_loss

    return loss_similarity


# data
batch_size = 2
data_loader = loader.PairedDataLoader("./data/train/mr_images", "./data/train/us_images",
                                      "./data/train/mr_labels", "./data/train/us_labels")
dataset = data_loader.get_dataset()
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)

# model
local_model = network.LocalModel(batch_size=batch_size, num_channel_initial=8)

# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
local_model.compile(optimizer, loss=loss_fn)
local_model.fit(dataset, epochs=10)
