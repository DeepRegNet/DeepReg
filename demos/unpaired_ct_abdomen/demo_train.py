import argparse

from deepreg.train import train

name = "unpaired_ct_abdomen"


def launch_training(method: str, test: bool = True):
    print(
        "The training can also be launched using the following command."
        "deepreg_train --gpu '0'"
        f"--config_path demos/{name}/{name}_{method}.yaml "
        f"--log_root demos/{name}"
        f"--log_dir logs_train/{method}"
    )

    log_root = f"demos/{name}"
    log_dir = f"logs_train/{method}"
    config_path = [f"demos/{name}/{name}_{method}.yaml"]
    if test:
        config_path.append("config/test/demo_unpaired_grouped.yaml")

    train(
        gpu="0",
        config_path=config_path,
        gpu_allow_growth=False,
        ckpt_path="",
        log_root=log_root,
        log_dir=log_dir,
    )


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        help="Training method, comb or unsup or weakly",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test",
        help="Execute the script for test purpose",
        dest="test",
        action="store_true",
    )
    parser.add_argument(
        "--no-test",
        help="Execute the script for non-test purpose",
        dest="test",
        action="store_false",
    )
    parser.set_defaults(test=True)
    args = parser.parse_args(args)
    assert args.method in [
        "comb",
        "unsup",
        "weakly",
    ], f"method should be comb or unsup or weakly, got {args.method}"

    launch_training(method=args.method, test=args.test)


if __name__ == "__main__":
    main()
