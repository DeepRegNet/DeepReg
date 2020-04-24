import os
from datetime import datetime

import click
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import deepreg.config.parser as config_parser
import deepreg.data.load as load
import deepreg.model.layer_util as layer_util
import deepreg.model.loss.label as label_loss
import deepreg.model.metric as metric
import deepreg.model.network as network
import deepreg.model.optimizer as opt


def predict(data_loader, dataset, fixed_grid_ref, model, save_dir):
    metric_map = dict()  # map[image_index][label_index][metric_name] = metric_value
    for i, (inputs, labels) in enumerate(dataset):
        # pred_fixed_label [batch, f_dim1, f_dim2, f_dim3]
        # moving_image     [batch, m_dim1, m_dim2, m_dim3]
        # fixed_image      [batch, f_dim1, f_dim2, f_dim3]
        # moving_label     [batch, m_dim1, m_dim2, m_dim3]
        # fixed_label      [batch, f_dim1, f_dim2, f_dim3]
        if hasattr(model, "ddf"):
            model_ddf = tf.keras.Model(inputs=model.inputs,
                                       outputs=model.outputs + [model.ddf])
            pred_fixed_label, ddf = model_ddf.predict(x=inputs)
        else:
            pred_fixed_label = model.predict(x=inputs)
            ddf = None

        moving_image, fixed_image, moving_label, indices = inputs
        fixed_label = labels
        num_samples = moving_image.shape[0]
        moving_depth = moving_image.shape[3]
        fixed_depth = fixed_image.shape[3]

        image_dir_format = save_dir + "/{image_dir:s}/label{label_index:d}"
        for sample_index in range(num_samples):
            image_index, label_index = data_loader.split_indices(indices[sample_index, :].numpy().astype(int).tolist())

            # save fixed
            image_dir = image_dir_format.format(image_dir=data_loader.image_index_to_dir(image_index),
                                                label_index=label_index)
            filename_format = image_dir + "/depth{depth_index:d}_{name:s}.png"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            for fixed_depth_index in range(fixed_depth):
                fixed_image_d = fixed_image[sample_index, :, :, fixed_depth_index]
                fixed_label_d = fixed_label[sample_index, :, :, fixed_depth_index]
                fixed_pred_d = pred_fixed_label[sample_index, :, :, fixed_depth_index]
                plt.imsave(
                    filename_format.format(depth_index=fixed_depth_index, name="fixed_image"),
                    fixed_image_d, cmap='gray')  # value range for h5 and nifti might be different
                plt.imsave(
                    filename_format.format(depth_index=fixed_depth_index, name="fixed_label"),
                    fixed_label_d, vmin=0, vmax=1, cmap='gray')
                plt.imsave(
                    filename_format.format(depth_index=fixed_depth_index, name="fixed_pred"),
                    fixed_pred_d, vmin=0, vmax=1, cmap='gray')

            # save moving
            image_dir = image_dir_format.format(image_dir=data_loader.image_index_to_dir(image_index),
                                                label_index=label_index)
            filename_format = image_dir + "/depth{depth_index:d}_{name:s}.png"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            for moving_depth_index in range(moving_depth):
                moving_image_d = moving_image[sample_index, :, :, moving_depth_index]
                moving_label_d = moving_label[sample_index, :, :, moving_depth_index]
                plt.imsave(
                    filename_format.format(depth_index=moving_depth_index, name="moving_image"),
                    moving_image_d, cmap='gray')  # value range for h5 and nifti might be different
                plt.imsave(
                    filename_format.format(depth_index=moving_depth_index, name="moving_label"),
                    moving_label_d, vmin=0, vmax=1, cmap='gray')

            # save ddf if exists
            if ddf is not None:
                image_dir = image_dir_format.format(image_dir=data_loader.image_index_to_dir(image_index),
                                                    label_index=label_index)
                filename_format = image_dir + "/depth{depth_index:d}_{name:s}.png"
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)
                for fixed_depth_index in range(fixed_depth):
                    ddf_d = ddf[sample_index, :, :, fixed_depth_index, :]  # [f_dim1, f_dim2,  3]
                    ddf_max, ddf_min = np.max(ddf_d), np.min(ddf_d)
                    ddf_d = (ddf_d - ddf_min) / (ddf_max - ddf_min)
                    plt.imsave(
                        filename_format.format(depth_index=fixed_depth_index, name="ddf"),
                        ddf_d)

            # calculate metric
            label = fixed_label[sample_index:(sample_index + 1), :, :, :]
            pred = pred_fixed_label[sample_index:(sample_index + 1), :, :, :]
            dice = label_loss.dice_score(y_true=label, y_pred=pred, binary=True)
            dist = label_loss.compute_centroid_distance(y_true=label, y_pred=pred,
                                                        grid=fixed_grid_ref)

            # save metric
            if image_index not in metric_map.keys():
                metric_map[image_index] = dict()
            assert label_index not in metric_map[image_index].keys()  # label should not be repeated
            metric_map[image_index][label_index] = dict(dice=dice.numpy()[0], dist=dist.numpy()[0])

    # print metric
    line_format = "{image_dir:s}, label {label_index:d}, dice {dice:.4f}, dist {dist:.4f}\n"
    with open(save_dir + "/metric.log", "w+") as f:
        for image_index in sorted(metric_map.keys()):
            for label_index in sorted(metric_map[image_index].keys()):
                f.write(line_format.format(image_dir=data_loader.image_index_to_dir(image_index),
                                           label_index=label_index,
                                           **metric_map[image_index][label_index]))


@click.command()
@click.option(
    "--gpu", "-g",
    help="GPU index",
    type=str,
    required=True,
)
@click.option(
    "--gpu_allow_growth/--not_gpu_allow_growth",
    help="Do not take all GPU memory",
    default=False,
    show_default=True)
@click.option(
    "--ckpt_path",
    help="Path of checkpoint to load",
    default="",
    show_default=True,
    type=str,
    required=True,
)
@click.option('--mode',
              help="Predict on train/valid/test data.",
              type=click.Choice(["tran", "valid", "test"],
                                case_sensitive=False),
              required=True)
@click.option(
    "--batch_size", "-b",
    help="Batch size",
    default=1,
    show_default=True,
    type=int,
)
@click.option(
    "--log",
    help="Name of log folder",
    default="",
    show_default=True,
    type=str,
)
def main(gpu, gpu_allow_growth, ckpt_path, mode, batch_size, log):
    # sanity check
    if not ckpt_path.endswith(".ckpt"):  # should be like log_folder/save/xxx.ckpt
        raise ValueError("checkpoint path should end with .ckpt")

    # env vars
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false" if gpu_allow_growth else "true"

    # load config
    config = config_parser.load("/".join(ckpt_path.split("/")[:-2]) + "/config.yaml")
    data_config = config["data"]
    if data_config["name"] == "mr_us":
        data_config["sample_label"]["train"] = "all"
        data_config["sample_label"]["test"] = "all"
    tf_data_config = config["tf"]["data"]
    tf_data_config["batch_size"] = batch_size
    tf_opt_config = config["tf"]["opt"]
    tf_model_config = config["tf"]["model"]
    tf_loss_config = config["tf"]["loss"]
    log_folder_name = log if log != "" else datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = config["log_dir"][:-1] if config["log_dir"][-1] == "/" else config["log_dir"]
    log_dir = log_dir + "/" + log_folder_name

    # data
    data_loader = load.get_data_loader(data_config, mode)
    dataset = data_loader.get_dataset_and_preprocess(training=False, repeat=False, **tf_data_config)

    # optimizer
    optimizer = opt.get_optimizer(tf_opt_config)

    # model
    model = network.build_model(moving_image_size=data_loader.moving_image_shape,
                                fixed_image_size=data_loader.fixed_image_shape,
                                index_size=data_loader.num_indices,
                                batch_size=tf_data_config["batch_size"],
                                tf_model_config=tf_model_config,
                                tf_loss_config=tf_loss_config)

    # metrics
    model.compile(optimizer=optimizer,
                  loss=label_loss.get_similarity_fn(config=tf_loss_config["similarity"]["label"]),
                  metrics=[metric.MeanDiceScore(),
                           metric.MeanCentroidDistance(grid_size=data_loader.fixed_image_shape)])

    # load weights
    model.load_weights(ckpt_path)

    # predict
    fixed_grid_ref = layer_util.get_reference_grid(grid_size=data_loader.fixed_image_shape)
    predict(data_loader=data_loader, dataset=dataset, fixed_grid_ref=fixed_grid_ref, model=model,
            save_dir=log_dir + "/test")


if __name__ == "__main__":
    main()
