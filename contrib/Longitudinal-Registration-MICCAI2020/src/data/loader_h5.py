import pickle as pkl
import random

import h5py
import numpy as np
import src.data.augmentation as aug
import tensorflow as tf


class H5DataLoader:
    def __init__(self, args, phase="train"):
        # load keys
        self.args = args
        self.data_aug = args.data_aug
        self.data_file = args.data_file
        self.key_file = args.key_file
        self.phase = phase
        self.key_pairs_list = self.get_key_pairs()
        self.moving_image_shape = args.image_shape
        self.fixed_image_shape = args.image_shape

    def get_generator(self):
        with h5py.File(self.data_file, "r") as df:
            self.key_pairs_list = self.get_key_pairs()  # re-shuffle again
            for image_index, (moving_key, fixed_key) in enumerate(self.key_pairs_list):
                moving_image, moving_label = df.get(moving_key)[()]
                fixed_image, fixed_label = df.get(fixed_key)[()]
                indices = np.asarray([image_index], dtype=np.float32)
                # print(moving_key, fixed_key)  # check samples
                yield ((moving_image, fixed_image, moving_label), fixed_label, indices)

    def _get_dataset(self):
        dataset = tf.data.Dataset.from_generator(
            generator=self.get_generator,
            output_types=((tf.float32, tf.float32, tf.float32), tf.float32, tf.float32),
            output_shapes=(
                (
                    self.moving_image_shape,
                    self.fixed_image_shape,
                    self.moving_image_shape,
                ),
                self.fixed_image_shape,
                1,
            ),
        )
        # self.key_pairs_list = self.get_key_pairs()
        return dataset

    def get_dataset(self, batch_size):
        dataset = self._get_dataset()
        dataset = dataset.batch(batch_size, drop_remainder=(self.phase == "train"))
        if (self.phase == "train") and self.data_aug:
            print("using data affine augmentation")
            affine_transform = aug.AffineTransformation3D(
                moving_image_size=self.moving_image_shape,
                fixed_image_size=self.fixed_image_shape,
                batch_size=batch_size,
            )
            dataset = dataset.map(affine_transform.transform)
            # pass
        return dataset

    def get_sorted_keys(self, filename):
        with h5py.File(filename, "r") as hf:
            return sorted(hf.keys())

    def get_key_pairs(self):
        with open(self.key_file, "rb") as f:
            key_dict = pkl.load(f)
        l = key_dict[self.phase]
        if self.phase == "train":
            if self.args.patient_cohort == "intra":
                l = self.__odd_even_shuffle__(l)
            elif self.args.patient_cohort == "inter":
                l = self.__get_inter_patient_pairs__(l)
            elif self.args.patient_cohort == "inter+intra":
                l1 = self.__odd_even_shuffle__(l)
                l2 = self.__get_inter_patient_pairs__(l)
                l3 = self.__inter_lock__(l1, l2)
                l = l3[: len(l)]
            else:
                print("wrong patient cohort.")
        return l

    def __get_inter_patient_pairs__(self, l):
        assert "random" in self.args.key_file, "key file should be random type"
        k = [i[0] for i in l]  # get all images
        l = [(i, j) for i in k for j in k]  # get all combinations
        l = [
            i for i in l if i[0].split("-")[0] != i[1].split("-")[0]
        ]  # exclude same patient
        random.shuffle(l)
        return l[: len(k)]  # get the same length as random ordered dataloader

    @staticmethod
    def __inter_lock__(l1, l2):
        new_list = []
        for a, b in zip(l1, l2):
            new_list.append(a)
            new_list.append(b)
        return new_list

    def __odd_even_shuffle__(self, l):
        even_list, odd_list, new_list = [], [], []
        for idx, i in enumerate(l):
            if (idx % 2) == 0:
                even_list.append(i)
            else:
                odd_list.append(i)
        random.shuffle(even_list)
        random.shuffle(odd_list)
        new_list = self.__inter_lock__(even_list, odd_list)
        return new_list
