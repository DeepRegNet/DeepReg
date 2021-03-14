import argparse
from datetime import datetime

from deepreg.predict import predict

name = "unpaired_us_prostate_cv"


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
    f"--ckpt_path demos/{name}/dataset/pretrained/ckpt-6000 "
    f"--log_root demos/{name} "
    "--log_dir logs_predict "
    "--save_png --mode test\n"
    "=========================================================\n"
    "\n\n\n\n\n"
)

log_root = f"demos/{name}"
log_dir = "logs_predict/" + datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_path = f"{log_root}/dataset/pretrained/ckpt-6000"
config_path = [f"{log_root}/{name}.yaml"]
if args.test:
    config_path.append("config/test/demo_unpaired_grouped.yaml")

predict(
    gpu="0",
    gpu_allow_growth=True,
    ckpt_path=ckpt_path,
    mode="test",
    batch_size=1,
    log_root=log_root,
    log_dir=log_dir,
    sample_label="all",
    config_path=config_path,
)
