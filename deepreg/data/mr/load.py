import deepreg.data.mr.loader_h5_ldmk as ldmk
import deepreg.data.mr.loader_h5_seg as seg


def get_data_loader(data_config, mode):
    data_order = data_config["data_order"] if mode == "train" else "forward"
    if data_config["data_type"] == "segmentation":
        return seg.H5SegmentationDataLoader(
            train_mode=mode, data_order=data_order, **data_config["segmentation"])
    elif data_config["data_type"] == "landmark":
        return ldmk.H5LandmarkDataLoader(
            train_mode=mode, data_order=data_order, **data_config["landmark"])
    else:
        raise ValueError("Unknown data type")
