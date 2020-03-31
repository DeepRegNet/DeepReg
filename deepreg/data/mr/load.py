import deepreg.data.mr.loader_h5_ldmk as ldmk_data_loader
import deepreg.data.mr.loader_h5_seg as seg_data_loader


def get_train_test_data_loader(data_type, data_config: dict):
    if data_type == "segmentation":
        data_loader_train = seg_data_loader.H5SegmentationDataLoader(
            train_mode="train", data_order="bidi", **data_config[data_type])
        data_loader_val = seg_data_loader.H5SegmentationDataLoader(
            train_mode="valid", data_order="forward", **data_config[data_type])
        data_loader_test = seg_data_loader.H5SegmentationDataLoader(
            train_mode="test", data_order="forward", **data_config[data_type])
    elif data_type == "landmark":
        data_loader_train = ldmk_data_loader.H5LandmarkDataLoader(
            train_mode="train", data_order="bidi", **data_config[data_type])
        data_loader_val = ldmk_data_loader.H5LandmarkDataLoader(
            train_mode="valid", data_order="forward", **data_config[data_type])
        data_loader_test = ldmk_data_loader.H5LandmarkDataLoader(
            train_mode="test", data_order="forward", **data_config[data_type])
    else:
        raise ValueError("Unknown data_type")
    return data_loader_train, data_loader_val, data_loader_test
