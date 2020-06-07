import os

from deepreg.data.mr_us.loader_h5 import H5DataLoader
from deepreg.data.mr_us.loader_nifti import NiftiDataLoader


def get_data_loader(data_config, mode):
    sample_label_train = data_config["sample_label"]["train"]
    sample_label_test = data_config["sample_label"]["test"]
    if data_config["format"] == "nifti":
        nifti_config = data_config["nifti"]
        data_dir = nifti_config["dir"][:-1] if nifti_config["dir"][-1] == "/" else nifti_config["dir"]
        load_into_memory = nifti_config["load_into_memory"]
        if data_config["tfrecord_dir"] != "" and data_config["tfrecord_dir"] != None:
            load_into_memory = False

        if mode == "train":
            moving_image_dir_train = data_dir + "/train/moving_images"
            fixed_image_dir_train = data_dir + "/train/fixed_images"
            moving_label_dir_train = data_dir + "/train/moving_labels"
            fixed_label_dir_train = data_dir + "/train/fixed_labels"

            return NiftiDataLoader(moving_image_dir=moving_image_dir_train,
                                   fixed_image_dir=fixed_image_dir_train,
                                   moving_label_dir=moving_label_dir_train,
                                   fixed_label_dir=fixed_label_dir_train,
                                   load_into_memory=load_into_memory,
                                   sample_label=sample_label_train,
                                   tfrecord_dir=os.path.join(data_config["tfrecord_dir"], "train"),
                                   )
        elif mode == "valid" or mode == "test":
            moving_image_dir_test = data_dir + "/test/moving_images"
            fixed_image_dir_test = data_dir + "/test/fixed_images"
            moving_label_dir_test = data_dir + "/test/moving_labels"
            fixed_label_dir_test = data_dir + "/test/fixed_labels"

            return NiftiDataLoader(moving_image_dir=moving_image_dir_test,
                                   fixed_image_dir=fixed_image_dir_test,
                                   moving_label_dir=moving_label_dir_test,
                                   fixed_label_dir=fixed_label_dir_test,
                                   load_into_memory=load_into_memory,
                                   sample_label=sample_label_test,
                                   tfrecord_dir=os.path.join(data_config["tfrecord_dir"], "test"),
                                   )
        else:
            raise ValueError("Unknown mode")
    elif data_config["format"] == "h5":
        h5_config = data_config["h5"]
        data_dir = h5_config["dir"][:-1] if h5_config["dir"][-1] == "/" else h5_config["dir"]

        moving_image_filename = data_dir + "/moving_images.h5"
        fixed_image_filename = data_dir + "/fixed_images.h5"
        moving_label_filename = data_dir + "/moving_labels.h5"
        fixed_label_filename = data_dir + "/fixed_labels.h5"

        if mode == "train":
            start_index_train = h5_config["train"]["start_image_index"]
            end_index_train = h5_config["train"]["end_image_index"]
            return H5DataLoader(moving_image_filename=moving_image_filename,
                                fixed_image_filename=fixed_image_filename,
                                moving_label_filename=moving_label_filename,
                                fixed_label_filename=fixed_label_filename,
                                start_image_index=start_index_train,
                                end_image_index=end_index_train,
                                sample_label=sample_label_train,
                                tfrecord_dir=os.path.join(data_config["tfrecord_dir"], "train"),
                                )

        elif mode == "valid" or mode == "test":
            start_index_test = h5_config["test"]["start_image_index"]
            end_index_test = h5_config["test"]["end_image_index"]
            return H5DataLoader(moving_image_filename=moving_image_filename,
                                fixed_image_filename=fixed_image_filename,
                                moving_label_filename=moving_label_filename,
                                fixed_label_filename=fixed_label_filename,
                                start_image_index=start_index_test,
                                end_image_index=end_index_test,
                                sample_label=sample_label_test,
                                tfrecord_dir=os.path.join(data_config["tfrecord_dir"], "test"),
                                )
    else:
        raise ValueError("Unknown option")
