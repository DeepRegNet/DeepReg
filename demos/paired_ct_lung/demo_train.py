import argparse
from datetime import datetime

from deepreg.train import train

name = "paired_ct_lung"

# parser is used to simplify testing
# please run the script with --no-test flag to ensure non-testing mode
# for instance:
# python script.py --no-test
parser = argparse.ArgumentParser()
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
args = parser.parse_args()

print(
    "\n\n\n\n\n"
    "=======================================================\n"
    "The training can also be launched using the following command.\n"
    "deepreg_train --gpu '0' "
    f"--config_path demos/{name}/{name}.yaml "
    f"--log_dir demos/{name} "
    "--exp_name logs_train\n"
    "=======================================================\n"
    "\n\n\n\n\n"
)

log_dir = f"demos/{name}"
exp_name = "logs_train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
config_path = [f"demos/{name}/{name}.yaml"]
if args.test:
    config_path.append("config/test/demo_paired.yaml")

train(
    gpu="0",
    config_path=config_path,
    gpu_allow_growth=True,
    ckpt_path="",
    log_dir=log_dir,
    exp_name=exp_name,
)
