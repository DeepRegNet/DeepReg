from deepreg.train import train

######## NOW WE DO THE TRAINING ########

gpu = ""
gpu_allow_growth = False
ckpt_path = ""
log_dir = "learn2reg_t2_unpaired_train_logs"
config_path = [
    r"deepreg/config/test/ddf.yaml",
    r"demos/unpaired_ct_lung/unpaired_ct_lung.yaml",
]
train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    log_dir=log_dir,
)
