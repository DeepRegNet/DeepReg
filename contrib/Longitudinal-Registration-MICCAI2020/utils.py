import os
import re

import matplotlib.pyplot as plt
import numpy as np


def gen_pred_imgs(ckpt_dir, pair_name, inputs, fixed_label, pred):
    slice_num = pred[0].shape[3]
    save_folder = os.path.join(
        os.path.dirname(os.path.dirname(ckpt_dir)),
        f"prediction-{os.path.basename(ckpt_dir)}",
        pair_name,
    )
    os.makedirs(save_folder, exist_ok=True)

    warped_moving_arr = pred[0].numpy()
    warped_label_arr = pred[1].numpy()
    fixed_label_arr = fixed_label.numpy()
    moving_arr = inputs[0].numpy()
    fixed_arr = inputs[1].numpy()
    moving_label = inputs[2].numpy()

    for s in range(slice_num):
        warped_moving_slice = warped_moving_arr[0, :, :, s, 0].T
        moving_slice = moving_arr[0, :, :, s].T
        fixed_slice = fixed_arr[0, :, :, s].T
        warped_moving_label_slice = warped_label_arr[0, :, :, s, 0].T
        moving_label_slice = moving_label[0, :, :, s].T
        fixed_label_slice = fixed_label_arr[0, :, :, s].T

        images = np.concatenate(
            [moving_slice, warped_moving_slice, fixed_slice], axis=1
        )
        labels = np.concatenate(
            [moving_label_slice, warped_moving_label_slice, fixed_label_slice], axis=1
        )
        combine = np.concatenate([images, labels], axis=0)
        fig_name = "{}-slice-{}.png".format(pair_name, s)
        plt.imsave(os.path.join(save_folder, fig_name), combine, cmap="gray")


def format_args(args):
    d = args.__dict__
    return "<br/>".join(["%s: %s" % (key, value) for (key, value) in d.items()])


def get_epoch_from_name(s):
    return int(re.findall("cp-(.*).ckpt.index", s)[0])


def get_exp_dir_and_ckpt(args):
    assert args.exp_name is not None, "must specify the experiment name"
    related_logs = [i for i in os.listdir(args.log_dir) if args.exp_name in i]
    assert len(related_logs) == 1, "experiment not unique"
    exp_dir = os.path.join(args.log_dir, related_logs[0])
    if args.continue_epoch == "latest":
        ckpt_folder = os.path.join(exp_dir, "checkpoint")
        ckpts = [
            get_epoch_from_name(s)
            for s in os.listdir(ckpt_folder)
            if s.endswith("index")
        ]
        ckpt_num = str(max(ckpts)).zfill(4)
    else:
        ckpt_num = args.continue_epoch.zfill(4)
    return (
        exp_dir,
        os.path.join(exp_dir, "checkpoint", f"cp-{ckpt_num}.ckpt"),
        int(ckpt_num) + 1,
    )


def sync2home(args):
    pass
