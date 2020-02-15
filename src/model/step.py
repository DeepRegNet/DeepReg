from enum import Enum

import numpy as np
import tensorflow as tf

import src.model.loss as loss
import src.model.metric as metric


class Mode(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


def init_metrics():  # this function should be consistent with the train/eval steps
    tb_names_test = dict(
        loss_sim="loss/similarity",
        loss_reg="loss/regularization",
        loss_total="loss/total",
        metric_dice="metric/dice",
        metric_dist="metric/centroid_distance",
        metric_dice_gland="metric/dice_gland",  # label index 0
        metric_dist_gland="metric/centroid_distance_gland",  # label index 0
    )
    tb_names_train = dict(
        **tb_names_test,
        opt_lr="opt/learning_rate",
    )
    metrics_train = metric.Metrics(tb_names=tb_names_train)
    metrics_test = metric.Metrics(tb_names=tb_names_test)
    return metrics_train, metrics_test


@tf.function
def step(model, inputs, labels, indices, fixed_grid_ref, tf_loss_config, optimizer, mode: Mode):
    """
    :param model:
    :param inputs:
    :param labels: [batch, f_dim1, f_dim2, f_dim3]
    :param indices: [batch, 2], first is image_index, second is label_index
    :param fixed_grid_ref:
    :param tf_loss_config:
    :param optimizer:
    :param mode: "train", "eval", "predict"
    :return:
    """
    with tf.GradientTape() as tape:
        preds = model(inputs=inputs, training=True)  # shape = [batch, f_dim1, f_dim2, f_dim3]
        if mode == Mode.PREDICT:
            return preds

        # loss
        loss_sim_value = loss.loss_similarity_fn(y_true=labels, y_pred=preds,
                                                 **tf_loss_config["similarity"])
        loss_reg_value = sum(model.losses)
        loss_total_value = loss_sim_value + loss_reg_value

        # metrics
        metric_dice_value = loss.binary_dice(y_true=labels, y_pred=preds)
        metric_dist_value = loss.compute_centroid_distance(y_true=labels, y_pred=preds, grid=fixed_grid_ref)

        # metrics for gland, label_index = 0
        mask = indices[:, 1] == 0  # shape = [batch, ]
        if tf.reduce_any(mask):
            masked_labels = tf.boolean_mask(labels, mask)
            masked_preds = tf.boolean_mask(preds, mask)
            metric_dice_gland_value = loss.binary_dice(y_true=masked_labels, y_pred=masked_preds)
            metric_dist_gland_value = loss.compute_centroid_distance(y_true=masked_labels, y_pred=masked_preds,
                                                                     grid=fixed_grid_ref)
        else:
            metric_dice_gland_value = np.nan
            metric_dist_gland_value = np.nan

    metric_value_dict = dict(
        loss_sim=loss_sim_value,
        loss_reg=loss_reg_value,
        loss_total=loss_total_value,
        metric_dice=metric_dice_value,
        metric_dist=metric_dist_value,
        metric_dice_gland=metric_dice_gland_value,
        metric_dist_gland=metric_dist_gland_value,
    )
    if mode == Mode.EVAL:
        return metric_value_dict

    # optimize
    grads = tape.gradient(loss_total_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # opt params
    opt_lr_value = optimizer._decayed_lr('float32')
    metric_value_dict["opt_lr"] = opt_lr_value

    if mode == Mode.TRAIN:
        return metric_value_dict


@tf.function
def train_step(model, inputs, labels, indices, fixed_grid_ref, tf_loss_config, optimizer):
    return step(model=model,
                inputs=inputs,
                labels=labels,
                indices=indices,
                fixed_grid_ref=fixed_grid_ref,
                tf_loss_config=tf_loss_config,
                optimizer=optimizer,
                mode=Mode.TRAIN)


@tf.function
def eval_step(model, inputs, labels, indices, fixed_grid_ref, tf_loss_config):
    return step(model=model,
                inputs=inputs,
                labels=labels,
                indices=indices,
                fixed_grid_ref=fixed_grid_ref,
                tf_loss_config=tf_loss_config,
                optimizer=None,
                mode=Mode.EVAL)


@tf.function
def predict_step(model, inputs):
    return step(model=model,
                inputs=inputs,
                labels=None,
                indices=None,
                fixed_grid_ref=None,
                tf_loss_config=None,
                optimizer=None,
                mode=Mode.PREDICT)
