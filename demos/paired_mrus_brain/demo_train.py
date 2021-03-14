import argparse
from datetime import datetime

from deepreg.train import train

name = "paired_mrus_brain"

# parser is used to simplify testing
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
parser.set_defaults(test=True)
args = parser.parse_args()


print(
    "\n\n\n\n\n"
    "=======================================================\n"
    "The training can also be launched using the following command.\n"
    "deepreg_train --gpu '0' "
    f"--config_path demos/{name}/{name}.yaml "
    f"--log_root demos/{name} "
    "--log_dir logs_train\n"
    "=======================================================\n"
    "\n\n\n\n\n"
)

log_root = f"demos/{name}"
log_dir = "logs_train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
config_path = [f"demos/{name}/{name}.yaml"]
if args.test:
    config_path.append("config/test/demo_paired.yaml")

train(
    gpu="0",
    config_path=config_path,
    gpu_allow_growth=True,
    ckpt_path="",
    log_root=log_root,
    log_dir=log_dir,
)
