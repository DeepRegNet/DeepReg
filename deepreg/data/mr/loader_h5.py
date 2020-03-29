import re

import h5py
import numpy as np
import tensorflow as tf

import deepreg.data.preprocess as preprocess
from deepreg.data.util import get_h5_sorted_keys

PATIENT_VISIT_KEY_FORMAT = "Patient{pid:d}-Visit{vid:d}"


class H5DataLoader:
    def __init__(self, train_mode, filename, valid_start, test_start, data_order):
        pid_vid_map = get_patient_visit_map(filename, train_mode, valid_start, test_start)
        self.pairs = get_key_pairs_from_map(pid_vid_map, data_order)
        self.filename = filename
        self.num_images = len(self.pairs)
        self.moving_image_shape = get_image_shape(filename)
        self.fixed_image_shape = self.moving_image_shape

    def get_generator(self):
        with h5py.File(self.filename, "r") as hf:
            for pair in self.pairs:
                moving_image, moving_label = hf.get(pair[0])[()]
                fixed_image, fixed_label = hf.get(pair[1])[()]
                indices = np.asarray([0, 0], dtype=np.float32)  # TODO nonsense values
                yield (moving_image, fixed_image, moving_label, indices), fixed_label

    def _get_dataset(self):
        return tf.data.Dataset.from_generator(
            generator=self.get_generator,
            output_types=((tf.float32, tf.float32, tf.float32, tf.float32), tf.float32),
            output_shapes=((self.moving_image_shape, self.fixed_image_shape, self.moving_image_shape, 2),
                           self.fixed_image_shape),
        )

    def get_dataset(self, training, batch_size, repeat: bool, shuffle_buffer_size):
        dataset = self._get_dataset()
        dataset = preprocess.preprocess(dataset=dataset,
                                        moving_image_shape=self.moving_image_shape,
                                        fixed_image_shape=self.fixed_image_shape,
                                        training=training,
                                        shuffle_buffer_size=shuffle_buffer_size,
                                        repeat=repeat,
                                        batch_size=batch_size)
        return dataset


def get_image_shape(filename):
    with h5py.File(filename, "r") as hf:
        keys = sorted([k for k in hf.keys()])
        sh = list(hf.get(keys[0]).shape)

        # sanity check
        # all samples have same 4d shape (2, dim1, dim2, dim3)
        assert len(sh) == 4
        for k in keys:
            assert sh == list(hf.get(k).shape)
    return sh[1:]


def _get_patient_visit_map(filename):
    pid_vid_map = dict()
    keys = get_h5_sorted_keys(filename)
    for k in keys:
        # find all numbers in the string
        # https://stackoverflow.com/questions/4289331/how-to-extract-numbers-from-a-string-in-python
        ids = [int(s) for s in re.findall(r'\d+', k)]
        if len(ids) != 2:
            raise ValueError("Key must be of form Patient%d-Visit%d")
        pid, vid = ids[0], ids[1]
        if pid in pid_vid_map:
            if vid in pid_vid_map[pid]:
                raise ValueError("Key repeated")
            else:
                pid_vid_map[pid].append(vid)
        else:
            pid_vid_map[pid] = [vid]
    # sort vids numerically
    for pid in pid_vid_map:
        pid_vid_map[pid] = sorted(pid_vid_map[pid])
    return pid_vid_map


def split_patient_visit_map(pid_vid_map: dict, valid_start, test_start):
    """
    split the map into train / valid / test
    :param pid_vid_map:
    :param valid_start: patient ID, included
    :param test_start: patient ID, included
    :return:
    """
    train, valid, test = dict(), dict(), dict()
    for pid, vids in pid_vid_map.items():
        if pid < valid_start:
            train[pid] = vids
        elif pid < test_start:
            valid[pid] = vids
        else:
            test[pid] = vids
    return train, valid, test


def get_patient_visit_map(filename, mode, valid_start, test_start):
    pid_vid_map = _get_patient_visit_map(filename)
    if mode == "train":
        pid_vid_map, _, _ = split_patient_visit_map(pid_vid_map, valid_start, test_start)
    elif mode == "valid":
        _, pid_vid_map, _ = split_patient_visit_map(pid_vid_map, valid_start, test_start)
    elif mode == "test":
        _, _, pid_vid_map = split_patient_visit_map(pid_vid_map, valid_start, test_start)
    else:
        raise ValueError("Unknown order")
    return pid_vid_map


def get_key_pairs_from_map(pid_vid_map: dict, order):
    # TODO always within one patient
    pairs = []
    for pid, vids in pid_vid_map.items():
        # must have at least two visits
        num_visits = len(vids)
        if num_visits <= 1:
            continue

        for j in range(num_visits):
            for i in range(j):
                # i < j
                ki = PATIENT_VISIT_KEY_FORMAT.format(pid=pid, vid=vids[i])
                kj = PATIENT_VISIT_KEY_FORMAT.format(pid=pid, vid=vids[j])
                if order == "forward":
                    pairs.append([ki, kj])
                elif order == "backward":
                    pairs.append([kj, ki])
                elif order == "bidi":  # bidirectional
                    pairs.append([ki, kj])
                    pairs.append([kj, ki])
    return pairs
