import argparse
from datetime import datetime

from deepreg.train import train

name = "unpaired_ct_abdomen"

# parser is used to simplify testing
# please run the script with --full flag to ensure non-testing mode
# for instance:
# python script.py --full
parser = argparse.ArgumentParser()
parser.add_argument(
    "--method",
    help="Training method, comb or unsup or weakly",
    type=str,
    required=True,
)
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
method = args.method

assert method in [
    "comb",
    "unsup",
    "weakly",
], f"method should be comb or unsup or weakly, got {method}"

print(
    "\n\n\n\n\n"
    "=======================================================\n"
    "The training can also be launched using the following command.\n"
    "deepreg_train --gpu '0' "
    f"--config_path demos/{name}/{name}_{method}.yaml "
    f"--log_dir demos/{name} "
    f"--log_dir logs_train/{method}\n"
    "=======================================================\n"
    "\n\n\n\n\n"
)

log_dir = f"demos/{name}"
exp_name = f"logs_train/{method}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
config_path = [f"demos/{name}/{name}_{method}.yaml"]
if args.test:
    config_path.append("config/test/demo_unpaired_grouped.yaml")

train(
    gpu="0",
    config_path=config_path,
    gpu_allow_growth=True,
    ckpt_path="",
    log_dir=log_dir,
    exp_name=exp_name,
)
