import os
import pickle as pkl
from glob import glob

import numpy as np
import src.data.loader_h5 as loader
import src.model.layer_util as layer_util
import src.model.network as network
import steps as steps
import tensorflow as tf
import utils as utils
from config import args

GPUs = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(GPUs[args.gpu], "GPU")
if GPUs and args.gpu_memory_control:
    for gpu in GPUs:
        tf.config.experimental.set_memory_growth(gpu, True)


assert (
    args.test_phase is not None
), "must specify the inference set, should be test/holdout"
data_loader_test = loader.H5DataLoader(args, phase=args.test_phase)
dataset_test = data_loader_test.get_dataset(batch_size=1)
local_model = network.build_model(
    moving_image_size=data_loader_test.moving_image_shape,
    fixed_image_size=data_loader_test.fixed_image_shape,
    batch_size=1,
    num_channel_initial=args.num_channel_initial,
    ddf_levels=args.ddf_levels,
    ddf_energy_type=args.ddf_energy_type,
)
fixed_grid_ref = layer_util.get_reference_grid(
    grid_size=data_loader_test.fixed_image_shape
)


log_dir, checkpoint_path, _ = utils.get_exp_dir_and_ckpt(args)
if args.test_phase == "test":
    checkpoint_paths = glob(os.path.join(log_dir, "checkpoint", "*.ckpt.index"))
    checkpoint_paths = [i.replace(".index", "") for i in checkpoint_paths]
    checkpoint_paths.sort()
    checkpoint_paths = checkpoint_paths[
        args.test_model_start_end[0] : args.test_model_start_end[1]
    ]
elif args.test_phase == "holdout":
    checkpoint_paths = [checkpoint_path]
else:
    checkpoint_paths = []
    print("must specify the set for inference.")
    raise NotImplementedError


inference_record = {}
ckpt_dice_mean, ckpt_dist_mean, ckpt_ssd_mean = [], [], []
ckpt_dice_std, ckpt_dist_std, ckpt_ssd_std = [], [], []

for checkpoint_path in checkpoint_paths:
    local_model.load_weights(checkpoint_path)
    centroid_distances = []
    dices = []
    ssds = []

    for step, (inputs, fixed_label, indices_test) in enumerate(dataset_test):
        dice, dist, ssd, pred = steps.valid_step(
            args_dict=args.__dict__,
            model=local_model,
            inputs=inputs,
            labels=fixed_label,
            fixed_grid_ref=fixed_grid_ref,
            return_type="prediction",
        )
        # print(step, dice, dist)
        centroid_distances.append(dist.numpy())
        dices.append(dice.numpy())
        ssds.append(ssd.numpy())
        if (args.test_phase == "holdout") and (args.test_gen_pred_imgs == 1):
            pair_name = (
                "-".join(data_loader_test.key_pairs_list[step])
                + f"-dice-{np.around(dice.numpy(), decimals=3)}-"
                f"cd{np.around(dist.numpy(), decimals=3)}"
            )
            utils.gen_pred_imgs(
                checkpoint_path,
                pair_name,
                inputs=inputs,
                fixed_label=fixed_label,
                pred=pred,
            )

    print(
        os.path.basename(checkpoint_path),
        f"Evaluation metrics, loss type {args.loss_type}, "
        f"dice: {np.mean(dices)}, {np.std(dices)}, "
        f"centroid_distances: {np.mean(centroid_distances)}, {np.std(centroid_distances)}, "
        f"ssds: {np.mean(ssds)}, {np.std(ssds)}",
    )

    ckpt_dice_mean.append(np.mean(dices))
    ckpt_dice_std.append(np.std(dices))
    ckpt_dist_mean.append(np.mean(centroid_distances))
    ckpt_dist_std.append(np.std(centroid_distances))
    ckpt_ssd_mean.append(np.mean(ssds))
    ckpt_ssd_std.append(np.std(ssds))

    inference_record[checkpoint_path] = {
        "dice": dices,
        "ssd": ssds,
        "centroid_distance": centroid_distances,
    }


index_dice, index_dist, index_ssd = (
    np.argmax(ckpt_dice_mean),
    np.argmin(ckpt_dist_mean),
    np.argmin(ckpt_ssd_mean),
)

print(
    "best dsc model:",
    checkpoint_paths[index_dice],
    ckpt_dice_mean[index_dice],
    ckpt_dice_std[index_dice],
)
print(
    "best TRE model:",
    checkpoint_paths[index_dist],
    ckpt_dist_mean[index_dist],
    ckpt_dist_std[index_dist],
)
print(
    "best ssd model:",
    checkpoint_paths[index_ssd],
    ckpt_ssd_mean[index_ssd],
    ckpt_ssd_std[index_ssd],
)


inference_record["dice_model"] = checkpoint_paths[index_dice]
inference_record["ssd_model"] = checkpoint_paths[index_ssd]
inference_record["cd_model"] = checkpoint_paths[index_dist]

inference_record["dice"] = [ckpt_dice_mean[index_dice], ckpt_dice_std[index_dice]]
inference_record["ssd"] = [ckpt_dist_mean[index_ssd], ckpt_dist_std[index_ssd]]
inference_record["cd"] = [ckpt_dist_mean[index_dist], ckpt_dist_std[index_dist]]


inference_record_path = args.test_phase + "-record.pkl"
if args.suffix != "":
    inference_record_path = args.suffix + "-" + inference_record_path
inference_record_path = os.path.join(
    os.path.dirname(os.path.dirname(checkpoint_path)), inference_record_path
)

with open(inference_record_path, "wb") as f:
    pkl.dump(inference_record, f)
