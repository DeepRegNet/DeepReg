import re

import h5py
import numpy as np

from deepreg.data.mr.loader import DataLoader
from deepreg.data.mr.util import get_image_shape, get_patient_map
from deepreg.data.util import get_h5_sorted_keys

PATIENT_VISIT_LANDMARK_KEY_FORMAT = "Patient{pid:d}-Visit{vid:d}-ldmark-{lid:d}"


class H5LandmarkDataLoader(DataLoader):
    def __init__(self, train_mode, data_order, filename, valid_start, test_start, tfrecord_dir):
        super(H5LandmarkDataLoader, self).__init__()
        pid_vid_lid_map = get_patient_map(get_patient_visit_landmark_map,
                                          filename, train_mode, valid_start, test_start)
        self.pairs = get_pid_vid_lid_pairs_from_map(pid_vid_lid_map, data_order)
        self.num_images = len(self.pairs)
        self.filename = filename
        self.moving_image_shape = get_image_shape(filename)
        self.fixed_image_shape = self.moving_image_shape
        self.num_indices = 5
        self.tfrecord_dir = tfrecord_dir

    def get_generator(self):
        with h5py.File(self.filename, "r") as hf:
            for pair in self.pairs:
                pid1, vid1, pid2, vid2, lids = pair
                for lid in lids:
                    k1 = PATIENT_VISIT_LANDMARK_KEY_FORMAT.format(pid=pid1, vid=vid1, lid=lid)
                    k2 = PATIENT_VISIT_LANDMARK_KEY_FORMAT.format(pid=pid2, vid=vid2, lid=lid)
                    moving_image, moving_label = hf.get(k1)[()]
                    fixed_image, fixed_label = hf.get(k2)[()]
                    indices = np.asarray([pid1, vid1, pid2, vid2,
                                          lid + 1,
                                          ], dtype=np.float32)
                    yield (moving_image, fixed_image, moving_label, indices), fixed_label


def get_patient_visit_landmark_map(filename):
    pid_vid_lid_map = dict()
    keys = get_h5_sorted_keys(filename)
    for k in keys:
        # find all numbers in the string
        # https://stackoverflow.com/questions/4289331/how-to-extract-numbers-from-a-string-in-python
        ids = [int(s) for s in re.findall(r'\d+', k)]
        if len(ids) != 3:
            raise ValueError("Key must be of form Patient%d-Visit%d-ldmark-%d")
        pid, vid, lid = ids[0], ids[1], ids[2]
        if pid in pid_vid_lid_map:
            if vid in pid_vid_lid_map[pid]:
                if lid in pid_vid_lid_map[pid][vid]:
                    raise ValueError("Key repeated")
                else:
                    pid_vid_lid_map[pid][vid].append(lid)
            else:
                pid_vid_lid_map[pid][vid] = [lid]
        else:
            pid_vid_lid_map[pid] = dict()

    # sort lids numerically
    for pid in pid_vid_lid_map:
        for vid in pid_vid_lid_map[pid]:
            pid_vid_lid_map[pid][vid] = sorted(pid_vid_lid_map[pid][vid])

    return pid_vid_lid_map


def get_pid_vid_lid_pairs_from_map(pid_vid_lid_map: dict, order):
    # TODO always within one patient
    pairs = []
    for pid, vid_map in pid_vid_lid_map.items():
        # must have at least two visits
        num_visits = len(vid_map)
        if num_visits <= 1:
            continue

        vids = list(vid_map.keys())
        for j in range(num_visits):
            for i in range(j):
                # i < j
                ki = [pid, vids[i]]
                kj = [pid, vids[j]]

                # get common landmarks
                lids_i = vid_map[vids[i]]
                lids_j = vid_map[vids[j]]
                lids = [x for x in lids_i if x in lids_j]
                if len(lids) == 0:  # no common landmarks
                    continue

                if order == "forward":
                    pairs.append(ki + kj + [lids])
                elif order == "backward":
                    pairs.append(kj + ki + [lids])
                elif order == "bidi":  # bidirectional
                    pairs.append(ki + kj + [lids])
                    pairs.append(kj + ki + [lids])

    return pairs
