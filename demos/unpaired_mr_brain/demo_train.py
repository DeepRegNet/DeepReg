import os

from deepreg.train import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

######## NOW WE DO THE TRAINING ########

gpu = "0"
gpu_allow_growth = False
ckpt_path = ""  # To load pre-trained weights
log_dir = "logs"
config_path = [r"demos/unpaired_mr_brain/unpaired_mr_brain.yaml"]
train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    log_dir=log_dir,
)
