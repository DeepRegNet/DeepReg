import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt

import src.config.parser as config_parser
import src.data.mr_us.loader as data_loader
import src.model.layer_util as layer_util
import src.model.loss.label as label_loss
import src.model.metric as metric
import src.model.network as network
import src.model.optimizer as opt


def predict(dataset, fixed_grid_ref, model, save_dir):
    metric_map = dict()  # map[image_index][label_index][metric_name] = metric_value
    for i, (inputs, labels) in enumerate(dataset):
        # pred_fixed_label [batch, f_dim1, f_dim2, f_dim3]
        # moving_image     [batch, m_dim1, m_dim2, m_dim3]
        # fixed_image      [batch, f_dim1, f_dim2, f_dim3]
        # moving_label     [batch, m_dim1, m_dim2, m_dim3]
        # fixed_label      [batch, f_dim1, f_dim2, f_dim3]
        pred_fixed_label = model.predict(
            x=inputs,
        )
        moving_image, fixed_image, moving_label, indices = inputs
        fixed_label = labels
        num_samples = moving_image.shape[0]
        moving_depth = moving_image.shape[3]
        fixed_depth = fixed_image.shape[3]

        image_dir_format = save_dir + "/image{image_index:d}/label{label_index:d}/{type_name:s}"
        for sample_index in range(num_samples):
            image_index, label_index = int(indices[sample_index, 0]), int(indices[sample_index, 1])

            # save fixed
            image_dir = image_dir_format.format(image_index=image_index, label_index=label_index, type_name="fixed")
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
            image_dir = image_dir_format.format(image_index=image_index, label_index=label_index, type_name="moving")
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
    line_format = "image {image_index:d}, label {label_index:d}, dice {dice:.4f}, dist {dist:.4f}\n"
    with open(save_dir + "/metric.log", "w+") as f:
        for image_index in sorted(metric_map.keys()):
            for label_index in sorted(metric_map[image_index].keys()):
                f.write(line_format.format(image_index=image_index, label_index=label_index,
                                           **metric_map[image_index][label_index]))


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--gpu", help="GPU index", required=True)
    parser.add_argument("-m", "--memory", action='store_true', help="take all GPU memory")
    parser.add_argument("--ckpt", help="Path of checkpoint", required=True)
    parser.add_argument("--bs", default=1, help="batch size")
    parser.add_argument("--train", action='store_true', help="predict on training set")
    parser.set_defaults(memory=True)
    parser.set_defaults(train=False)
    args = parser.parse_args()

    # env vars
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "false" if args.memory else "true"

    checkpoint_path = args.ckpt  # would be like log_folder/save/xxx.ckpt
    if not checkpoint_path.endswith(".ckpt"):
        raise ValueError("checkpoint path should end with .ckpt")

    # load config
    config = config_parser.load("/".join(checkpoint_path.split("/")[:-2]) + "/config.yaml")
    data_config = config["data"]
    data_config["sample_label"]["train"] = "all"
    data_config["sample_label"]["test"] = "all"
    tf_data_config = config["tf"]["data"]
    tf_data_config["batch_size"] = int(args.bs)
    tf_opt_config = config["tf"]["opt"]
    tf_model_config = config["tf"]["model"]
    tf_loss_config = config["tf"]["loss"]
    log_dir = config["log_dir"][:-1] if config["log_dir"][-1] == "/" else config["log_dir"]
    log_dir = log_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # data
    data_loader_train, data_loader_val = data_loader.get_train_test_dataset(data_config)
    dataset_train = data_loader_train.get_dataset(training=True, repeat=False, **tf_data_config)
    dataset_val = data_loader_val.get_dataset(training=False, repeat=False, **tf_data_config)

    # optimizer
    optimizer = opt.get_optimizer(tf_opt_config)

    # model
    reg_model = network.build_model(moving_image_size=data_loader_train.moving_image_shape,
                                    fixed_image_size=data_loader_train.fixed_image_shape,
                                    batch_size=tf_data_config["batch_size"],
                                    tf_model_config=tf_model_config,
                                    tf_loss_config=tf_loss_config)

    # metrics
    reg_model.compile(optimizer=optimizer,
                      loss=label_loss.get_similarity_fn(config=tf_loss_config["similarity"]["label"]),
                      metrics=[metric.MeanDiceScore(),
                               metric.MeanCentroidDistance(grid_size=data_loader_train.fixed_image_shape)])

    # load weights
    reg_model.load_weights(checkpoint_path)

    # predict
    fixed_grid_ref = layer_util.get_reference_grid(grid_size=data_loader_train.fixed_image_shape)
    predict(dataset=dataset_val, fixed_grid_ref=fixed_grid_ref, model=reg_model, save_dir=log_dir + "/test")
    if args.train:
        predict(dataset=dataset_train, fixed_grid_ref=fixed_grid_ref, model=reg_model, save_dir=log_dir + "/train")
