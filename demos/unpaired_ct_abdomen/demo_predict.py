# pylint: disable=line-too-long
import argparse
from datetime import datetime

from deepreg.predict import predict

name = "unpaired_ct_abdomen"
ckpt_index_dict = {"comb": 5000, "unsup": 5000, "weakly": 5000}

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
parser.set_defaults(test=False)
args = parser.parse_args()
method = args.method
assert method in [
    "comb",
    "unsup",
    "weakly",
], f"method should be comb or unsup or weakly, got {method}"

ckpt_index = ckpt_index_dict[method]
print(
    "\n\n\n\n\n"
    "=========================================================\n"
    "The prediction can also be launched using the following command.\n"
    "deepreg_predict --gpu '' "
    f"--config_path demos/{name}/{name}_{method}.yaml "
    f"--ckpt_path demos/{name}/dataset/pretrained/{method}/ckpt-{ckpt_index} "
    f"--log_dir demos/{name} "
    f"--log_dir logs_predict/{method} "
    "--save_png --split test\n"
    "=========================================================\n"
    "\n\n\n\n\n"
)

log_dir = f"demos/{name}"
exp_name = f"logs_predict/{method}/" + datetime.now().strftime("%Y%m%d-%H%M%S")
ckpt_path = f"{log_dir}/dataset/pretrained/{method}/ckpt-{ckpt_index}"
config_path = [f"{log_dir}/{name}_{method}.yaml"]
if args.test:
    config_path.append("config/test/demo_unpaired_grouped.yaml")

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
