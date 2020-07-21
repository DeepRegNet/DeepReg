from deepreg.train import train
import tensorflow as tf

######## NOW WE DO THE TRAINING ########

tf.get_logger().setLevel(3)
tf.keras.backend.clear_session()

gpu = ""
gpu_allow_growth = False
ckpt_path = ""
log_dir = "learn2reg_t1_paired_train_logs"
config_path = [
    r"deepreg/config/test/ddf.yaml",
    r"demos/paired_mr_us_brain/paired_mr_us_brain.yaml",
]
train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path=ckpt_path,
    log_dir=log_dir,
)