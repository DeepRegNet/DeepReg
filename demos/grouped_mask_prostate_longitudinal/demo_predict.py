from deepreg.predict import predict

print(
    "The prediction can also be launched using the following command."
    "deepreg_predict --gpu "
    " --config_path demos/grouped_mask_prostate_longitudinal/grouped_mask_prostate_longitudinal.yaml --ckpt_path demos/grouped_mask_prostate_longitudinal/dataset/pre-trained/weights-epoch500.ckpt --save_png --mode test"
)


log_dir = "unpaired_ct_abdomen_log_comb"
ckpt_path = (
    "demos/grouped_mask_prostate_longitudinal/dataset/pre-trained/weights-epoch500.ckpt"
)
config_path = (
    "demos/grouped_mask_prostate_longitudinal/grouped_mask_prostate_longitudinal.yaml"
)

predict(
    gpu="0",
    gpu_allow_growth=False,
    ckpt_path=ckpt_path,
    mode="test",
    batch_size=1,
    log_dir=log_dir,
    sample_label="all",
    config_path=config_path,
)
