import argparse
from datetime import datetime

from deepreg.train import train

name = "grouped_mask_prostate_longitudinal"

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
parser.add_argument(
    "--remote",
    help="Use gpu romotely (on Cluster)",
    dest="remote",
    action="store_true",
)
parser.set_defaults(test=True, remote=False)
args = parser.parse_args()

print(
    "\n\n\n\n\n"
    "=======================================================\n"
    "The training can also be launched using the following command.\n"
    "deepreg_train --gpu '0' "
    f"--config_path demos/{name}/{name}.yaml "
    f"--log_dir demos/{name} "
    "--exp_name logs_train\n"
    "If using remote GPU, change to --gpu 'all' \n"
    "=======================================================\n"
    "\n\n\n\n\n"
)

log_dir = f"demos/{name}"
exp_name = "logs_train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
config_path = [f"demos/{name}/{name}.yaml"]
gpu = "0"
gpu_allow_growth = True
if args.test:
    config_path.append("config/test/demo_unpaired_grouped.yaml")
if args.remote:
    gpu = "all"
    gpu_allow_growth = False
train(
    gpu=gpu,
    config_path=config_path,
    gpu_allow_growth=gpu_allow_growth,
    ckpt_path="",
    log_dir=log_dir,
    exp_name=exp_name,
)
