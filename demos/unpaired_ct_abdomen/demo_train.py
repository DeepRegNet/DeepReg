from deepreg.train import train

# Training

gpu = ""
gpu_allow_growth = False
ckpt_path = ""
log_dir = "unpaired_ct_abdomen_log"
config_path = [
    r"deepreg/config/test/ddf.yaml",
    r"demos/unpaired_ct_abdomen/unpaired_ct_abdomen.yaml",
]
train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    log_dir=log_dir,
)
