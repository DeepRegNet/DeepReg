from src.data.loader_h5 import H5DataLoader
from src.data.loader_nifti import NiftiDataLoader


def get_train_test_dataset(data_config: dict):
    if data_config["format"] == "nifti":
        data_config = data_config["nifti"]
        data_dir = data_config["dir"][:-1] if data_config["dir"][-1] == "/" else data_config["dir"]
        load_into_memory = data_config["load_into_memory"]
        sample_label_train = data_config["train"]["sample_label"]
        sample_label_test = data_config["test"]["sample_label"]

        moving_image_dir_train = data_dir + "/train/moving_images"
        fixed_image_dir_train = data_dir + "/train/fixed_images"
        moving_label_dir_train = data_dir + "/train/moving_labels"
        fixed_label_dir_train = data_dir + "/train/fixed_labels"

        moving_image_dir_test = data_dir + "/test/moving_images"
        fixed_image_dir_test = data_dir + "/test/fixed_images"
        moving_label_dir_test = data_dir + "/test/moving_labels"
        fixed_label_dir_test = data_dir + "/test/fixed_labels"

        data_loader_train = NiftiDataLoader(moving_image_dir=moving_image_dir_train,
                                            fixed_image_dir=fixed_image_dir_train,
                                            moving_label_dir=moving_label_dir_train,
                                            fixed_label_dir=fixed_label_dir_train,
                                            load_into_memory=load_into_memory,
                                            sample_label=sample_label_train)

        data_loader_test = NiftiDataLoader(moving_image_dir=moving_image_dir_test,
                                           fixed_image_dir=fixed_image_dir_test,
                                           moving_label_dir=moving_label_dir_test,
                                           fixed_label_dir=fixed_label_dir_test,
                                           load_into_memory=load_into_memory,
                                           sample_label=sample_label_test)
    elif data_config["format"] == "h5":
        data_config = data_config["h5"]
        data_dir = data_config["dir"][:-1] if data_config["dir"][-1] == "/" else data_config["dir"]

        moving_image_filename = data_dir + "/moving_images.h5"
        fixed_image_filename = data_dir + "/fixed_images.h5"
        moving_label_filename = data_dir + "/moving_labels.h5"
        fixed_label_filename = data_dir + "/fixed_labels.h5"

        seed = data_config["seed"]
        shuffle = data_config["shuffle_image"]
        start_index_train = data_config["train"]["start_image_index"]
        end_index_train = data_config["train"]["end_image_index"]
        sample_label_train = data_config["train"]["sample_label"]
        start_index_test = data_config["test"]["start_image_index"]
        end_index_test = data_config["test"]["end_image_index"]
        sample_label_test = data_config["test"]["sample_label"]

        data_loader_train = H5DataLoader(moving_image_filename=moving_image_filename,
                                         fixed_image_filename=fixed_image_filename,
                                         moving_label_filename=moving_label_filename,
                                         fixed_label_filename=fixed_label_filename,
                                         seed=seed,
                                         shuffle_image=shuffle,
                                         start_image_index=start_index_train,
                                         end_image_index=end_index_train,
                                         sample_label=sample_label_train)

        data_loader_test = H5DataLoader(moving_image_filename=moving_image_filename,
                                        fixed_image_filename=fixed_image_filename,
                                        moving_label_filename=moving_label_filename,
                                        fixed_label_filename=fixed_label_filename,
                                        seed=seed,
                                        shuffle_image=shuffle,
                                        start_image_index=start_index_test,
                                        end_image_index=end_index_test,
                                        sample_label=sample_label_test)
    else:
        raise ValueError("Unknown option")
    return data_loader_train, data_loader_test
