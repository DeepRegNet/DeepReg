from datetime import datetime

from deepreg.predict import predict

name = "unpaired_ct_lung"

print(
    "\n\n\n\n\n"
    "========================================================="
    "The prediction can also be launched using the following command.\n"
    "deepreg_predict --gpu '' "
    f"--config_path demos/{name}/{name}.yaml "
    f"--ckpt_path demos/{name}/dataset/pretrained/learn2reg_t2_unpaired_train_logs/save/weights-epoch1500.ckpt "
    f"--log_root demos/{name} "
    "--log_dir logs_predict "
    "--save_png --mode test\n"
    "=========================================================\n"
    "\n\n\n\n\n"
)

log_root = f"demos/{name}"
log_dir = "logs_predict/" + datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_path = f"{log_root}/dataset/pretrained/learn2reg_t2_unpaired_train_logs/save/weights-epoch1500.ckpt"
config_path = f"{log_root}/{name}.yaml"

predict(
    gpu="0",
    gpu_allow_growth=False,
    ckpt_path=ckpt_path,
    mode="test",
    batch_size=1,
    log_root=log_root,
    log_dir=log_dir,
    sample_label="all",
    config_path=config_path,
)
