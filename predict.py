import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt

import src.config.parser as config_parser
import src.data.loader as data_loader
import src.model.layer_util as layer_util
import src.model.loss as loss
import src.model.network as network
from src.model import step as steps


def predict(dataset, fixed_grid_ref, model, save_dir):
    metric_map = dict()  # map[image_index][label_index][metric_name] = metric_value

    for step, (inputs, labels, indices) in enumerate(dataset):
        pred_fixed_label = steps.predict_step(model=model, inputs=inputs)
        pred_fixed_label = pred_fixed_label[..., 0]

        # moving_image [bs, m_dim1, m_dim2, m_dim3]
        # fixed_image  [bs, f_dim1, f_dim2, f_dim3]
        # moving_label [bs, m_dim1, m_dim2, m_dim3]
        # fixed_label  [bs, f_dim1, f_dim2, f_dim3]
        # pred_moving_label [bs, f_dim1, f_dim2, f_dim3]
        moving_image, fixed_image, moving_label = inputs  # shape [bs, dim1, dim2, dim3], [bs, dim1, dim2, dim3],
        fixed_label = labels
        num_samples = moving_image.shape[0]
        fixed_depth = fixed_image.shape[3]

        image_dir_format = save_dir + "/image{image_index:d}/label{label_index:d}"
        for sample_index in range(num_samples):
            # save prediction
            image_index, label_index = int(indices[sample_index, 0]), int(indices[sample_index, 1])
            image_dir = image_dir_format.format(image_index=image_index, label_index=label_index)
            filename_format = image_dir + "/depth{depth_index:d}_{name:s}.png"
            if not os.path.exists(image_dir):
                os.makedirs(image_dir)
            for depth_index in range(fixed_depth):
                image = fixed_image[sample_index, :, :, depth_index]
                label = fixed_label[sample_index, :, :, depth_index]
                pred = pred_fixed_label[sample_index, :, :, depth_index]
                plt.imsave(
                    filename_format.format(depth_index=depth_index, name="image"),
                    image, vmin=0, vmax=255, cmap='gray')
                plt.imsave(
                    filename_format.format(depth_index=depth_index, name="label"),
                    label, vmin=0, vmax=1, cmap='gray')
                plt.imsave(
                    filename_format.format(depth_index=depth_index, name="pred"),
                    pred, vmin=0, vmax=1, cmap='gray')

            # calculate metric
            label = fixed_label[sample_index:(sample_index + 1), ...]
            pred = pred_fixed_label[sample_index:(sample_index + 1), ...]
            dice = loss.binary_dice(y_true=label, y_pred=pred)
            dist = loss.compute_centroid_distance(y_true=label, y_pred=pred, grid=fixed_grid_ref)

            # save metric
            if image_index not in metric_map.keys():
                metric_map[image_index] = dict()
            assert label_index not in metric_map[image_index].keys()  # label should not be repeated
            metric_map[image_index][label_index] = dict(dice=dice.numpy()[0], dist=dist.numpy())

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
    parser.add_argument('-p', '--path', help='Path of checkpoint', required=True)
    args = parser.parse_args()

    checkpoint_path = args.path
    if not checkpoint_path.endswith(".ckpt"):
        raise ValueError("checkpoint path should end with .ckpt")

    # load config
    config = config_parser.load_default()
    data_config = config["data"]
    tf_data_config = config["tf"]["data"]
    tf_model_config = config["tf"]["model"]
    tf_loss_config = config["tf"]["loss"]
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if config["tf"]["TF_FORCE_GPU_ALLOW_GROWTH"] else "false"
    log_dir = config["log_dir"][:-1] if config["log_dir"][-1] == "/" else config["log_dir"]
    log_dir = log_dir + "/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    # overwrite config
    data_config[data_config["format"]]["train"]["sample_label"] = False
    data_config[data_config["format"]]["test"]["sample_label"] = False

    # data
    data_loader_train, data_loader_test = data_loader.get_train_test_dataset(data_config)
    dataset_train = data_loader_train.get_dataset(training=False, **tf_data_config)
    dataset_test = data_loader_test.get_dataset(training=False, **tf_data_config)
    fixed_grid_ref = layer_util.get_reference_grid(grid_size=data_loader_train.fixed_image_shape)

    # model
    reg_model = network.build_model(moving_image_size=data_loader_test.moving_image_shape,
                                    fixed_image_size=data_loader_test.fixed_image_shape,
                                    batch_size=tf_data_config["batch_size"],
                                    tf_model_config=tf_model_config,
                                    tf_loss_config=tf_loss_config)
    reg_model.load_weights(checkpoint_path)

    # predict
    predict(dataset=dataset_train, fixed_grid_ref=fixed_grid_ref, model=reg_model, save_dir=log_dir + "/train")
    predict(dataset=dataset_test, fixed_grid_ref=fixed_grid_ref, model=reg_model, save_dir=log_dir + "/test")
