import argparse
from datetime import datetime

from deepreg.predict import predict

name = "paired_mrus_prostate"

# parser is used to simplify testing, by default it is not used
# please run the script with --full flag to ensure non-testing mode
# for instance:
# python script.py --full
parser = argparse.ArgumentParser()
parser.add_argument(
    "--test",
    help="Execute the script with reduced image size for test purpose.",
    dest="test",
    action="store_true",
)
parser.add_argument(
    "--full",
    help="Execute the script with full configuration.",
    dest="test",
    action="store_false",
)
parser.set_defaults(test=False)
args = parser.parse_args()

print(
    "\n\n\n\n\n"
    "=========================================================\n"
    "The prediction can also be launched using the following command.\n"
    "deepreg_predict --gpu '' "
    f"--config_path demos/{name}/{name}.yaml "
    f"--ckpt_path demos/{name}/dataset/pretrained/ckpt-5000 "
    f"--log_dir demos/{name} "
    "--exp_name logs_predict "
    "--save_png --split test\n"
    "=========================================================\n"
    "\n\n\n\n\n"
)

log_dir = f"demos/{name}"
exp_name = "logs_predict/" + datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_path = f"{log_dir}/dataset/pretrained/ckpt-5000"
config_path = [f"{log_dir}/{name}.yaml"]
if args.test:
    config_path.append("config/test/demo_paired.yaml")

predict(
    gpu="0",
    gpu_allow_growth=True,
    ckpt_path=ckpt_path,
    split="test",
    batch_size=1,
    log_dir=log_dir,
    exp_name=exp_name,
    config_path=config_path,
)
