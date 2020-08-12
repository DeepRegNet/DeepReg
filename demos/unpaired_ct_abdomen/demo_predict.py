import os

from deepreg.predict import predict

log_dir = "unpaired_ct_abdomen_log"
ckpt_path = os.path.join("logs", log_dir, "save", "weights-epoch2.ckpt")

gpu = ""
gpu_allow_growth = False

config_path = [
    r"deepreg/config/test/ddf.yaml",
    r"demos/unpaired_ct_abdomen/unpaired_ct_abdomen.yaml",
]

predict(
    gpu=gpu,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    mode="test",
    batch_size=1,
    log_dir=log_dir,
    sample_label="all",
    config_path=config_path
)