import deepreg.data.mr.loader_h5_ldmk as ldmk
import deepreg.data.mr.loader_h5_seg as seg


def get_data_loader(data_config, mode):
    """
    landmarks are used only for test
    :param data_config:
    :param mode:
    :return:
    """
    data_order = data_config["data_order"] if mode == "train" else "forward"
    if mode != "test":
        return seg.H5SegmentationDataLoader(
            train_mode=mode, data_order=data_order, **data_config["segmentation"])
    else:
        return ldmk.H5LandmarkDataLoader(
            train_mode=mode, data_order=data_order, **data_config["landmark"])
