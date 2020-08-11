from deepreg.train import train

######## NOW WE DO THE TRAINING ########

gpu = "0"
gpu_allow_growth = False
ckpt_path = ""
log_dir = "learn2reg_t1_paired_train_logs"
config_path = [
    r"deepreg/config/test/ddf.yaml",
    r"demos/paired_mr_us_brain/paired_mr_us_brain.yaml",
]
save_png = True
train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    log_dir=log_dir,
    save_png=save_png,
)
