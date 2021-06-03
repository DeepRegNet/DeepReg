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
    "--dvf", help="using dvf to train the model", dest="dvf", action="store_true"
)
parser.set_defaults(test=True, dvf=False)
args = parser.parse_args()

print(
    "\n\n\n\n\n"
    "=======================================================\n"
    "The ddf training can also be launched using the following command.\n"
    "deepreg_train --gpu '0' "
    f"--config_path demos/{name}/{name}_ddf.yaml "
    f"--log_dir demos/{name} "
    "--exp_name logs_train\n"
    "=======================================================\n"
    "\n\n\n\n\n"
)

log_dir = f"demos/{name}"
exp_name = "logs_train/" + datetime.now().strftime("%Y%m%d-%H%M%S")
if args.dvf:
    config_path = [f"demos/{name}/{name}_dvf.yaml"]
else:
    config_path = [f"demos/{name}/{name}_ddf.yaml"]
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
