import os

from deepreg.data.mr_us.loader_h5 import H5DataLoader
from deepreg.data.mr_us.loader_nifti import NiftiDataLoader


def get_data_loader(data_config, mode):
    sample_label_train = data_config["sample_label"]["train"]
    sample_label_test = data_config["sample_label"]["test"]
    if mode == "valid":
        mode = "test"
    if data_config["format"] == "nifti":
        nifti_config = data_config["nifti"]
        data_dir = nifti_config["dir"][:-1] if nifti_config["dir"][-1] == "/" else nifti_config["dir"]
        load_into_memory = nifti_config["load_into_memory"]
        if data_config["tfrecord_dir"] != "" and data_config["tfrecord_dir"] != None:
            load_into_memory = False

        moving_image_dir_train = data_dir + "/%s/moving_images" % mode
        fixed_image_dir_train = data_dir + "/%s/fixed_images" % mode
        moving_label_dir_train = data_dir + "/%s/moving_labels" % mode
        fixed_label_dir_train = data_dir + "/%s/fixed_labels" % mode
        tfrecord_dir = None
        if data_config["tfrecord_dir"] != "" and data_config["tfrecord_dir"] is not None:
            tfrecord_dir = os.path.join(data_config["tfrecord_dir"], mode)
        return NiftiDataLoader(moving_image_dir=moving_image_dir_train,
                               fixed_image_dir=fixed_image_dir_train,
                               moving_label_dir=moving_label_dir_train,
                               fixed_label_dir=fixed_label_dir_train,
                               load_into_memory=load_into_memory,
                               sample_label=sample_label_train,
                               tfrecord_dir=tfrecord_dir,
                               )
    elif data_config["format"] == "h5":
        h5_config = data_config["h5"]
        data_dir = h5_config["dir"][:-1] if h5_config["dir"][-1] == "/" else h5_config["dir"]

        moving_image_filename = data_dir + "/moving_images.h5"
        fixed_image_filename = data_dir + "/fixed_images.h5"
        moving_label_filename = data_dir + "/moving_labels.h5"
        fixed_label_filename = data_dir + "/fixed_labels.h5"
        tfrecord_dir = None
        if data_config["tfrecord_dir"] != "" and data_config["tfrecord_dir"] is not None:
            tfrecord_dir = os.path.join(data_config["tfrecord_dir"], mode)

        start_index_train = h5_config[mode]["start_image_index"]
        end_index_train = h5_config[mode]["end_image_index"]
        return H5DataLoader(moving_image_filename=moving_image_filename,
                            fixed_image_filename=fixed_image_filename,
                            moving_label_filename=moving_label_filename,
                            fixed_label_filename=fixed_label_filename,
                            start_image_index=start_index_train,
                            end_image_index=end_index_train,
                            sample_label=sample_label_train,
                            tfrecord_dir=tfrecord_dir,
                            )
    else:
        raise ValueError("Unknown option")
