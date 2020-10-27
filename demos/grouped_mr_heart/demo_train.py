import argparse
from datetime import datetime

from deepreg.train import train

name = "grouped_mr_heart"

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
    "The training can also be launched using the following command."
    "deepreg_train --gpu '0' "
    f"--config_path demos/{name}/{name}.yaml "
    f"--log_root demos/{name} "
    "--log_dir logs_train"
)

log_root = f"demos/{name}"
log_dir = "logs_train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
config_path = [f"demos/{name}/{name}.yaml"]
if args.test:
    config_path.append("config/test/demo_unpaired_grouped.yaml")

train(
    gpu="0",
    config_path=config_path,
    gpu_allow_growth=False,
    ckpt_path="",
    log_root=log_root,
    log_dir=log_dir,
)
