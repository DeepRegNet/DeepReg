import argparse

from deepreg.predict import predict

name = "unpaired_ct_abdomen"
ckpt_index_dict = {"comb": 2000, "unsup": 5000, "weakly": 2250}


def launch_prediction(method: str):
    ckpt_index = ckpt_index_dict[method]
    print(
        "The prediction can also be launched using the following command."
        "deepreg_predict --gpu ''"
        f"--config_path demos/{name}/{name}_{method}.yaml "
        f"--ckpt_path demos/{name}/dataset/pretrained/{method}/weights-epoch{ckpt_index}.ckpt "
        f"--log_root demos/{name} "
        f"--log_dir logs_predict/{method}"
        "--save_png --mode test"
    )

    log_root = f"demos/{name}"
    log_dir = f"logs_predict/{method}"
    ckpt_path = f"{log_root}/dataset/pretrained/{method}/weights-epoch{ckpt_index}.ckpt"
    config_path = f"{log_root}/{name}_{method}.yaml"

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


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        help="Training method, comb or unsup or weakly",
        type=str,
        required=True,
    )
    args = parser.parse_args(args)
    assert args.method in [
        "comb",
        "unsup",
        "weakly",
    ], f"method should be comb or unsup or weakly, got {args.method}"

    launch_prediction(method=args.method)


if __name__ == "__main__":
    main()
