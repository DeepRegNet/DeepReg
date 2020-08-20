from deepreg.train import train

######## NOW WE DO THE TRAINING ########

gpu = ""
gpu_allow_growth = False
ckpt_path = ""
log_dir = "learn2reg_t2_paired_train_logs"
config_path = [
    r"demos/paired_ct_lung/paired_ct_lung_train.yaml",
    r"demos/paired_ct_lung/paired_ct_lung.yaml",
]
train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    log_dir=log_dir,
)
