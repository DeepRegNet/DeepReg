import os

from deepreg.train import train

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
######## NOW WE DO THE TRAINING ########

gpu = "0"
gpu_allow_growth = False
ckpt_path = ""  # To load pre-trained weights
log_dir = "/home/acasamitjana/Repositories/DeepReg/demos/mr-brain-t4/learn2reg_t4_unpaired_train_logs"
config_path = [
    "/home/acasamitjana/Repositories/DeepReg/demos/mr-brain-t4/mr_brain_t4.yaml"
]
train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    log_dir=log_dir,
)
