import os
import random

import nibabel as nib
import numpy as np
import tensorflow as tf

import src.data.augmentation as aug


class DataLoader:
    def __init__(self, image_dir, label_dir):
        # sanity check
        if image_dir is None:
            raise ValueError("Image directory path must not be None.")
        if image_dir == label_dir:
            raise ValueError("Image and label directory path is the same.")

        # load data into memory
        images, image_fnames = self.load_data(image_dir)
        labels, label_fnames = self.load_data(label_dir)

        # sanity check
        for i in range(len(image_fnames)):
            if len(images[i].shape) != 3:
                raise ValueError("The %d-th image's dimesion is not 3: %d." % (i, len(images[i].shape)))
        if labels is not None:
            if len(image_fnames) != len(label_fnames):
                raise ValueError(
                    "The number of images (%d) and labels (%d) do not match." % (len(image_fnames), len(label_fnames)))
            for i in range(len(image_fnames)):
                if image_fnames[i] != label_fnames[i]:
                    raise ValueError("The %d-th image and label's file name do not match: %s (image), %s (label)." % (
                        i, image_fnames[i], label_fnames[i]))
                if len(labels[i].shape) not in [3, 4]:
                    raise ValueError("The %d-th label's dimesion is not 3 or 4: %d." % (i, len(labels[i].shape)))
                if images[i].shape != labels[i].shape[:3]:
                    raise ValueError("The %d-th image and label's shape do not match: %s (image), %s (label)." % (
                        i, images[i].shape, labels[i].shape))

        # save data
        self.images = images
        self.labels = labels
        self.image_fnames = image_fnames

    @staticmethod
    def load_data(dir_name):
        if dir_name is None:
            return None, None
        file_names = os.listdir(dir_name)
        file_names.sort()
        data = [np.asarray(nib.load(os.path.join(dir_name, fname)).dataobj, dtype=np.float32) for fname in file_names]
        return data, file_names

    def get_image(self, i):
        return self.images[i]

    def get_label(self, i, label_index=None):
        label = self.labels[i]
        if len(label.shape) == 4:
            if label_index is None:
                label_index = random.randrange(label.shape[3])
            label = label[..., label_index]
        return label, label_index


class PairedDataLoader:
    def __init__(self, moving_image_dir, fixed_image_dir, moving_label_dir, fixed_label_dir):
        # sanity check
        if (moving_label_dir is None) != (fixed_label_dir is None):
            raise ValueError("The label paths should be both None or provided.")

        # load data
        loader_moving = DataLoader(moving_image_dir, moving_label_dir)
        loader_fixed = DataLoader(fixed_image_dir, fixed_label_dir)

        # sanity check
        if len(loader_moving.image_fnames) != len(loader_fixed.image_fnames):
            raise ValueError("The number of moving images (%d) and fixed images (%d) do not match." % (
                len(loader_moving.image_fnames), len(loader_fixed.image_fnames)))
        for i in range(len(loader_moving.image_fnames)):
            if loader_moving.image_fnames[i] != loader_fixed.image_fnames[i]:
                raise ValueError(
                    "The %d-th moving and fixed image's file name do not match: %s (image), %s (label)." % (
                        i, loader_moving.image_fnames[i], loader_fixed.image_fnames[i]))

        # save
        self.loader_moving = loader_moving
        self.loader_fixed = loader_fixed
        self.moving_image_shape = list(loader_moving.get_image(0).shape)
        self.fixed_image_shape = list(loader_fixed.get_image(0).shape)

    def get_generator(self):
        """
        For both moving and fixed, the image is always provided, but the label might not be provided,
        if the label is not provided, it only generates (moving_image, fixed_image) pairs,
        otherwise, generates (moving_image, fixed_image, moving_label), fixed_label pairs.
        """
        num_samples = len(self.loader_moving.images)
        for sample_index in range(num_samples):
            moving_image = self.loader_moving.get_image(sample_index)
            fixed_image = self.loader_fixed.get_image(sample_index)
            if self.loader_moving.labels is None:
                raise NotImplementedError
            else:
                moving_label, label_index = self.loader_moving.get_label(sample_index)
                fixed_label, _ = self.loader_fixed.get_label(sample_index, label_index)
                label_index = -1 if label_index is None else label_index
                indices = np.asarray([sample_index, label_index], dtype=np.float32)
                yield ((moving_image, fixed_image, moving_label), fixed_label, indices)

    def _get_dataset(self):
        if self.loader_moving.labels is None:
            raise NotImplementedError
        else:
            dataset = tf.data.Dataset.from_generator(generator=self.get_generator,
                                                     output_types=(
                                                         (tf.float32, tf.float32, tf.float32), tf.float32, tf.float32),
                                                     output_shapes=((self.moving_image_shape,
                                                                     self.fixed_image_shape,
                                                                     self.moving_image_shape),
                                                                    self.fixed_image_shape,
                                                                    2,
                                                                    ))

        return dataset

    def get_dataset(self, batch_size, training, dataset_shuffle_buffer_size):
        dataset = self._get_dataset()
        if training:
            dataset = dataset.shuffle(buffer_size=dataset_shuffle_buffer_size)
        dataset = dataset.batch(batch_size, drop_remainder=training)
        if training:
            # TODO add cropping, but crop first or rotation first?
            affine_transform = aug.AffineTransformation3D(moving_image_size=self.moving_image_shape,
                                                          fixed_image_size=self.fixed_label_shape,
                                                          batch_size=batch_size)
            dataset = dataset.map(affine_transform.transform)
        return dataset
