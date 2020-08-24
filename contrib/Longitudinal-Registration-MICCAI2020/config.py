import argparse

import utils as utils

parser = argparse.ArgumentParser()
# common options
parser.add_argument(
    "--exp_name", default=None, type=str, help="experiment name you want to add."
)
parser.add_argument(
    "--data_file",
    default="./data/ImageWithSeg-0.7-0.7-0.7-64-64-51-SimpleNorm-RmOut.h5",
    type=str,
    help="the h5 file data",
)
parser.add_argument(
    "--key_file",
    default="./data/exp1-key-random.pkl",
    type=str,
    help="the key index data",
)
parser.add_argument(
    "--image_shape", default=[128, 128, 102], nargs="+", type=int, help="image shape"
)
# Training options
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate.")
parser.add_argument("--w_ncc", default=1.0, type=float, help="the weight of ncc loss")
parser.add_argument("--w_ssd", default=1.0, type=float, help="the weight of ssd loss")
parser.add_argument(
    "--w_bde", default=10.0, type=float, help="the weight of bending energy loss"
)
parser.add_argument(
    "--w_mmd",
    default=1.0,
    type=float,
    help="the weight of maximum mean discrepancy loss",
)
parser.add_argument("--w_dce", default=1.0, type=float, help="the weight of dice loss")
parser.add_argument(
    "--sigmas",
    default=[
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1.0,
        5.0,
        10.0,
        15.0,
        20.0,
        25.0,
        30.0,
        35.0,
        100.0,
        1e3,
        1e4,
        1e5,
        1e6,
    ],
    nargs="+",
    type=float,
    help="sigmas for the gaussian kernels in mmd loss",
)
parser.add_argument(
    "--patient_cohort",
    default="intra",
    type=str,
    help="intra-patient type or inter-patient type",
)
parser.add_argument(
    "--loss_type",
    default="ssd",
    type=str,
    help="the type of the similarity loss, ncc/ssd",
)
parser.add_argument(
    "--data_aug", default=1, type=int, help="use data affine augmentation or not"
)
parser.add_argument(
    "--continue_epoch",
    default="-1",
    type=str,
    help="continue training from a certain ckpt",
)  ##
parser.add_argument(
    "--batch_size", default=4, type=int, help="The number of batch size."
)
parser.add_argument("--gpu", default=0, type=int, help="id of gpu")
parser.add_argument("--epochs", default=800, type=int, help="The number of iterations.")
parser.add_argument(
    "--log_dir",
    default="./logs",
    type=str,
    help="Folder for logs, checkpoints and predictions",
)
parser.add_argument("--save_period", default=5, type=int, help="save period")
parser.add_argument(
    "--num_channel_initial",
    default=16,
    type=int,
    help="number of channels in first layer.",
)
parser.add_argument(
    "--dataset_shuffle_buffer_size",
    default=64,
    type=int,
    help="dataset_shuffle_buffer_size",
)
parser.add_argument(
    "--ddf_levels",
    default=[0, 1, 2, 3, 4],
    nargs="+",
    type=int,
    help="ddf levels, numbers should be <= 4",
)
parser.add_argument(
    "--ddf_energy_type",
    default="bending",
    type=str,
    help="could be gradient-l2, gradient-l1, bending",
)
parser.add_argument(
    "--gpu_memory_control",
    default=0,
    type=int,
    help="use all or minimum of memory, 0 - not use",
)
# Testing options
parser.add_argument(
    "--test_gen_pred_imgs",
    default=0,
    type=int,
    help="generate the prediction images when inference?",
)
parser.add_argument(
    "--test_phase",
    default="test",
    type=str,
    help="use test or holdout set in the inference?",
)
parser.add_argument(
    "--test_mode",
    default=0,
    type=int,
    help="set a flag for test to avoid some computation",
)
parser.add_argument(
    "--test_model_start_end",
    default=[0, 200],
    nargs="+",
    type=int,
    help="the range of model you wanna test.",
)
parser.add_argument(
    "--test_before_reg", default=0, type=int, help="test without registration"
)
parser.add_argument(
    "--suffix", default="", type=str, help="the suffix for the results record"
)

args = parser.parse_args()

assert args.batch_size > 1, "batch size must larger than 1"
