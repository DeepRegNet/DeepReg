import shutil

import numpy as np
import tensorflow as tf

from deepreg.callback import CheckpointManagerCallback


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
    old_model = Net()
    old_model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss=tf.keras.losses.MSE)
    old_callback = CheckpointManagerCallback(
        model=old_model,
        directory="./test/unit/ckpt_old",
        period=5,
    )
    old_model.fit(
        x=toy_dataset(), epochs=10, steps_per_epoch=10, callbacks=[old_callback]
    )

    # create new model and restore old_model checkpoint
    new_model = Net()
    new_model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss=tf.keras.losses.MSE)
    new_callback = CheckpointManagerCallback(
        model=old_model, directory="./test/unit/ckpt_new"
    )
    new_callback.restore(save_path="./test/unit/ckpt_old/ckpt-10")

    # check equal
    new_callback._manager.save()
    old_reader = tf.train.load_checkpoint("./test/unit/ckpt_old/ckpt-10")
    new_reader = tf.train.load_checkpoint("./test/unit/ckpt_new")
    for k in old_reader.get_variable_to_shape_map().keys():
        if "save_counter" not in k:
            equal = np.array(old_reader.get_tensor(k)) == np.array(
                new_reader.get_tensor(k)
            )
            assert equal.all(), k

    # remove temporary ckpt directories
    shutil.rmtree("./test/unit/ckpt_old")
    shutil.rmtree("./test/unit/ckpt_new")


# if __name__ == '__main__':
#     test_restore_CheckpointManagerCallback()
