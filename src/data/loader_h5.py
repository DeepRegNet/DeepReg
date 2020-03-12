import random

import h5py
import numpy as np

from src.data.loader_basic import BasicDataLoader

SKIPPED_KEYS = ["num_important", "num_labels"]  # keys in label h5 file


class H5DataLoader(BasicDataLoader):
    def __init__(self,
                 moving_image_filename, fixed_image_filename, moving_label_filename, fixed_label_filename,
                 start_image_index, end_image_index):
        """

        :param moving_image_filename:
        :param fixed_image_filename:
        :param moving_label_filename:
        :param fixed_label_filename:
        :param seed: random seed
        :param shuffle_image:
        :param start_image_index: included
        :param end_image_index: excluded
        :param sample_label: sample label for each image sample or iterate label
        """
        super(H5DataLoader, self).__init__()
        # load image_keys
        moving_image_label_map = get_image_label_map(image_filename=moving_image_filename,
                                                     label_filename=moving_label_filename)
        fixed_image_label_map = get_image_label_map(image_filename=fixed_image_filename,
                                                    label_filename=fixed_label_filename)

        # sanity check
        # two key dicts is the same
        for k in moving_image_label_map:
            assert moving_image_label_map[k] == fixed_image_label_map[k]

        moving_image_shape = get_image_shape(moving_image_filename)
        fixed_image_shape = get_image_shape(fixed_image_filename)
        moving_label_shape = get_image_shape(moving_label_filename)
        fixed_label_shape = get_image_shape(fixed_label_filename)

        # sanity check
        # image and label have same shape
        assert moving_image_shape == moving_label_shape
        assert fixed_image_shape == fixed_label_shape

        # take specific data
        image_keys = sorted(moving_image_label_map)  # get sorted image_keys
        assert start_image_index >= 0 and end_image_index <= len(image_keys)
        image_keys = image_keys[start_image_index:end_image_index]

        # save variables
        self.image_label_map = moving_image_label_map
        self.image_keys = image_keys

        self.moving_image_filename = moving_image_filename
        self.fixed_image_filename = fixed_image_filename
        self.moving_label_filename = moving_label_filename
        self.fixed_label_filename = fixed_label_filename

        self.moving_image_shape = moving_image_shape
        self.fixed_image_shape = fixed_image_shape

    def get_generator(self):
        """
        For both moving and fixed, the image is always provided, but the label might not be provided,
        if the label is not provided, it only generates (moving_image, fixed_image) pairs,
        otherwise, generates (moving_image, fixed_image, moving_label), fixed_label pairs.
        """
        with h5py.File(self.moving_image_filename, "r") as hf_moving_image:
            with h5py.File(self.moving_label_filename, "r") as hf_moving_label:
                with h5py.File(self.fixed_image_filename, "r") as hf_fixed_image:
                    with h5py.File(self.fixed_label_filename, "r") as hf_fixed_label:
                        image_keys = self.image_keys
                        for image_index, image_key in enumerate(image_keys):
                            label_indices = [random.randrange(len(self.image_label_map[image_key]))]

                            moving_image = hf_moving_image.get(image_key)[()]
                            fixed_image = hf_fixed_image.get(image_key)[()]
                            for label_index in label_indices:
                                label_key = self.image_label_map[image_key][label_index]
                                moving_label = hf_moving_label.get(label_key)[()]
                                fixed_label = hf_fixed_label.get(label_key)[()]

                                indices = np.asarray([image_index, label_index], dtype=np.float32)
                                yield (moving_image, fixed_image, moving_label), fixed_label, indices


def get_sorted_keys(filename):
    with h5py.File(filename, "r") as hf:
        return sorted(hf.keys())


def get_image_label_map(image_filename, label_filename):
    """
    for images, the keys of h5 are like ["case000000", "case000001",  ...]
    for labels, the keys of h5 are like ["case000000_bin000", "case000000_bin001", ...,
                                         "case000001_bin000", "case000001_bin001", ...,
                                         ...,
                                         "num_important", "num_labels"]

    :param image_filename:
    :param label_filename:
    :return:
    """

    # load keys
    image_keys = get_sorted_keys(filename=image_filename)
    label_keys = get_sorted_keys(filename=label_filename)

    # build dictionary
    image_label_map = dict()  # map image key to label key
    for label_key in label_keys:
        image_key = label_key.split("_bin")[0]
        if image_key not in SKIPPED_KEYS:
            assert image_key in image_keys
            if image_key not in image_label_map.keys():
                image_label_map[image_key] = []
            image_label_map[image_key].append(label_key)  # no need to sort afterwards as label_keys are sorted

    # sanity check
    # all samples have labels
    assert sorted(image_label_map.keys()) == image_keys

    for image_key in image_keys:
        image_label_map[image_key] = sorted(image_label_map[image_key])

    return image_label_map


def get_image_shape(filename):
    with h5py.File(filename, "r") as hf:
        keys = sorted([k for k in hf.keys() if k not in SKIPPED_KEYS])
        sh = list(hf.get(keys[0]).shape)

        # sanity check
        # all samples have same 3d shape
        assert len(sh) == 3
        for k in keys:
            assert sh == list(hf.get(k).shape)
    return sh
