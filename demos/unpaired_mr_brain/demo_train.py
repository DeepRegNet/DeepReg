import os

from deepreg.train import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpu = "0"
gpu_allow_growth = False
ckpt_path = ""  # To load pre-trained weights
config_path = [r"demos/unpaired_mr_brain/unpaired_mr_brain.yaml"]

# log_dir: this log dir points to the downloaded log dir. Change it for other experiments.
log_dir = "learn2reg_t4_unpaired_train_logs"

train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    log_dir=log_dir,
)
