import os

from deepreg.predict import predict

######## PREDICTION ########


log_dir = "learn2reg_t1_paired_train_logs"
ckpt_path = os.path.join("logs", log_dir, "save", "weights-epoch2.ckpt")
config_path = "logs/learn2reg_t1_paired_train_logs/config.yaml"

gpu = ""
gpu_allow_growth = False
predict(
    gpu=gpu,
    gpu_allow_growth=gpu_allow_growth,
    config_path=config_path,
    ckpt_path=ckpt_path,
    mode="test",
    batch_size=1,
    log_dir=log_dir,
    sample_label="all",
)

# the numerical metrics are saved in the logs directory specified
