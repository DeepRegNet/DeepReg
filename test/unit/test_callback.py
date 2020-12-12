import shutil

import numpy as np
import tensorflow as tf

from deepreg.callback import build_callbacks, restore_model


def test_restore_CheckpointManagerCallback():
    """
    testing restore CheckpointManagerCallback
    """

    # toy model
    class Net(tf.keras.Model):
        """A simple linear model."""

        def __init__(self):
            super(Net, self).__init__()
            self.l1 = tf.keras.layers.Dense(5)

        def __call__(self, x, training=False):
            return self.l1(x)

    # toy dataset
    def toy_dataset():
        inputs = tf.range(10.0)[:, None]
        labels = inputs * 5.0 + tf.range(5.0)[None, :]
        return tf.data.Dataset.from_tensor_slices((inputs, labels)).repeat().batch(2)

    # train old_model and save
    if len(tf.config.list_physical_devices("gpu")) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:  # use default strategy
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        old_model = Net()
        old_optimizer = tf.keras.optimizers.Adam(0.1)
    old_model.compile(optimizer=old_optimizer, loss=tf.keras.losses.MSE)
    old_callbacks = build_callbacks(
        model=old_model,
        dataset=toy_dataset(),
        log_dir="./test/unit/old",
        histogram_freq=1,
        save_period=5,
    )
    old_model.fit(
        x=toy_dataset(), epochs=10, steps_per_epoch=10, callbacks=old_callbacks
    )

    # create new model and restore old_model checkpoint
    with strategy.scope():
        new_model = Net()
        new_optimizer = tf.keras.optimizers.Adam(0.1)
    new_model.compile(optimizer=new_optimizer, loss=tf.keras.losses.MSE)
    new_callbacks = build_callbacks(
        model=new_model,
        dataset=toy_dataset(),
        log_dir="./test/unit/new",
        histogram_freq=1,
        save_period=5,
    )
    initial_epoch = restore_model(
        new_callbacks, ckpt_path="./test/unit/old/save/ckpt-10"
    )

    # check equal
    new_callbacks[1]._manager.save(0)
    old_reader = tf.train.load_checkpoint("./test/unit/old/save/ckpt-10")
    new_reader = tf.train.load_checkpoint("./test/unit/new/save")
    for k in old_reader.get_variable_to_shape_map().keys():
        if "save_counter" not in k and "_CHECKPOINTABLE_OBJECT_GRAPH" not in k:
            equal = np.array(old_reader.get_tensor(k)) == np.array(
                new_reader.get_tensor(k)
            )
            assert np.all(equal), "{} fail to restore !".format(k)

    new_model.fit(
        x=toy_dataset(),
        initial_epoch=initial_epoch,
        epochs=20,
        steps_per_epoch=10,
        callbacks=new_callbacks,
    )

    # remove temporary ckpt directories
    shutil.rmtree("./test/unit/old")
    shutil.rmtree("./test/unit/new")
