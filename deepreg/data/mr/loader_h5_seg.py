import re

import h5py
import numpy as np

from deepreg.data.loader_basic import BasicDataLoader
from deepreg.data.mr.util import get_image_shape, get_patient_map
from deepreg.data.util import get_h5_sorted_keys

PATIENT_VISIT_KEY_FORMAT = "Patient{pid:d}-Visit{vid:d}"


class H5SegmentationDataLoader(BasicDataLoader):
    def __init__(self, train_mode, data_order, filename, valid_start, test_start):
        super(H5SegmentationDataLoader, self).__init__()
        pid_vid_map = get_patient_map(get_patient_visit_map,
                                      filename, train_mode, valid_start, test_start)
        self.pairs = get_pid_vid_pairs_from_map(pid_vid_map, data_order)
        self.num_images = len(self.pairs)
        self.filename = filename
        self.moving_image_shape = get_image_shape(filename)
        self.fixed_image_shape = self.moving_image_shape
        self.num_indices = 5

    def get_generator(self):
        with h5py.File(self.filename, "r") as hf:
            for pair in self.pairs:
                pid1, vid1, pid2, vid2 = pair
                k1 = PATIENT_VISIT_KEY_FORMAT.format(pid=pid1, vid=vid1)
                k2 = PATIENT_VISIT_KEY_FORMAT.format(pid=pid2, vid=vid2)
                moving_image, moving_label = hf.get(k1)[()]
                fixed_image, fixed_label = hf.get(k2)[()]
                indices = np.asarray([pid1, vid1, pid2, vid2,
                                      0,  # means it's segmentation
                                      ], dtype=np.float32)
                yield (moving_image, fixed_image, moving_label, indices), fixed_label


def get_patient_visit_map(filename):
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


def get_pid_vid_pairs_from_map(pid_vid_map: dict, order):
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
                ki = [pid, vids[i]]
                kj = [pid, vids[j]]
                if order == "forward":
                    pairs.append(ki + kj)
                elif order == "backward":
                    pairs.append(kj + ki)
                elif order == "bidi":  # bidirectional
                    pairs.append(ki + kj)
                    pairs.append(kj + ki)
    return pairs
