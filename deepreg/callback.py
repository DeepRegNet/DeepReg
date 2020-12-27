import tensorflow as tf


def build_checkpoint_manager_callback(
    model, dataset, log_dir: str, save_period: int, ckpt_path: str
):
    """
    Function to prepare callbacks for training.

    :param log_dir: directory of logs
    :param histogram_freq: save the histogram every X epochs
    :param save_period: save the checkpoint every X epochs
    :param ckpt_path: path to restore ckpt
    :return: a list of callbacks
    """
    # fit the model for 1 step to initialise optimiser arguments as trackable Variables
    model.fit(
        x=dataset,
        steps_per_epoch=1,
        epochs=1,
    )
    checkpoint_manager_callback = CheckpointManagerCallback(
        model, log_dir + "/save", period=save_period
    )
    if ckpt_path:
        initial_epoch = int(ckpt_path.split("-")[-1])
        checkpoint_manager_callback.restore(ckpt_path)
    else:
        initial_epoch = 0
    return checkpoint_manager_callback, initial_epoch


def restore_model(callbacks, ckpt_path):
    if ckpt_path:
        initial_epoch = int(ckpt_path.split("-")[-1])
        callbacks[-1].restore(ckpt_path)
    else:
        initial_epoch = 0
    return initial_epoch


class CheckpointManagerCallback(tf.keras.callbacks.Callback):
    def __init__(
        self, model, directory, period: int = 1, save_on_train_end: bool = True
    ):
        """
        Callback wrapping `tf.train.CheckpointManager`.

        :param model: model
        :param directory: directory to store the checkpoints
        :param period: save the checkpoint every X epochs
        :param save_on_train_end: save the checkpoint as the training ends
        """
        super(CheckpointManagerCallback, self).__init__()
        self._directory = directory

        self._checkpoint = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
        self._manager = tf.train.CheckpointManager(
            checkpoint=self._checkpoint, directory=self._directory, max_to_keep=None
        )
        self._period = period
        self._save_on_train_end = save_on_train_end
        self._restored = False
        self._epoch_count = None
        self._last_save = None

    def _on_begin(self):
        if not self._restored:
            self.restore()

    def restore(self, save_path=None):
        if save_path is None:
            save_path = self._manager.latest_checkpoint
        self._checkpoint.restore(save_path)
        self._restored = True

    def on_train_begin(self, logs=None):
        self._on_begin()

    def on_epoch_end(self, epoch, logs=None):
        epochs_finished = epoch + 1
        self._epoch_count = epochs_finished
        if epochs_finished % self._period == 0:
            self._save()

    def on_train_end(self, logs=None):
        if self._save_on_train_end:
            self._save()

    def _save(self):
        """
        checkpoint saved as './{}/ckpt-{}'.format(self._directory, self._epoch_count)
        """
        if self._last_save != self._epoch_count:
            self._manager.save(checkpoint_number=self._epoch_count)
            self._last_save = self._epoch_count
