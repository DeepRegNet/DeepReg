from deepreg.train import train

######## TRAINING ########

gpu = "0"
gpu_allow_growth = False
ckpt_path = ""
log_dir = "learn2reg_t1_paired_train_logs"
config_path = [
    r"demos/paired_mrus_brain/paired_mrus_brain_train.yaml",
    r"demos/paired_mrus_brain/paired_mrus_brain.yaml",
]
train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    log_dir=log_dir,
)
