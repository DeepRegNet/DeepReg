from deepreg.train import train


def test_train():
    gpu = ""
    config_path = "deepreg/config/mr_us_ddf.yaml"
    gpu_allow_growth = False
    ckpt_path = ""
    log = "test_train"
    train(gpu, config_path, gpu_allow_growth, ckpt_path, log)
