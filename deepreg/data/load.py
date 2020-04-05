import deepreg.data.mr.load as load_mr
import deepreg.data.mr_us.load as load_mr_us


def get_data_loader(data_config, mode):
    if data_config["name"] == "mr_us":
        return load_mr_us.get_data_loader(data_config, mode)
    elif data_config["name"] == "mr":
        return load_mr.get_data_loader(data_config, mode)
    else:
        raise ValueError("Unknown data name")
