import tensorflow as tf

import src.model.loss as loss
import src.model.metric as metric


def init_metrics():  # this function should be consistent with the train/eval steps
    tb_names_test = dict(
        loss_sim="loss/similarity",
        loss_reg="loss/regularization",
        loss_total="loss/total",
        metric_dice="metric/dice",
        metric_dist="metric/centroid_distance",
    )
    tb_names_train = dict(
        **tb_names_test,
        opt_lr="opt/learning_rate",
    )
    metrics_train = metric.Metrics(tb_names=tb_names_train)
    metrics_test = metric.Metrics(tb_names=tb_names_test)
    return metrics_train, metrics_test


@tf.function
def train_step(model, optimizer, inputs, labels, fixed_grid_ref, tf_loss_config):
    # forward
    with tf.GradientTape() as tape:
        predictions = model(inputs=inputs, training=True)

        # loss
        loss_sim_value = loss.loss_similarity_fn(y_true=labels, y_pred=predictions,
                                                 **tf_loss_config["similarity"])
        loss_reg_value = sum(model.losses)
        loss_total_value = loss_sim_value + loss_reg_value

        # metrics
        metric_dice_value = loss.binary_dice(labels, predictions)
        metric_dist_value = loss.compute_centroid_distance(labels, predictions, fixed_grid_ref)

    # optimize
    grads = tape.gradient(loss_total_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # opt params
    opt_lr_value = optimizer._decayed_lr('float32')

    return dict(
        loss_sim=loss_sim_value,
        loss_reg=loss_reg_value,
        loss_total=loss_total_value,
        metric_dice=metric_dice_value,
        metric_dist=metric_dist_value,
        opt_lr=opt_lr_value,
    )


@tf.function
def eval_step(model, inputs, labels, fixed_grid_ref, tf_loss_config):
    # forward
    predictions = model(inputs=inputs, training=False)

    # loss
    loss_sim_value = loss.loss_similarity_fn(y_true=labels, y_pred=predictions, **tf_loss_config["similarity"])
    loss_reg_value = sum(model.losses)
    loss_total_value = loss_sim_value + loss_reg_value

    # metrics
    metric_dice_value = loss.binary_dice(labels, predictions)
    metric_dist_value = loss.compute_centroid_distance(labels, predictions, fixed_grid_ref)

    return dict(
        loss_sim=loss_sim_value,
        loss_reg=loss_reg_value,
        loss_total=loss_total_value,
        metric_dice=metric_dice_value,
        metric_dist=metric_dist_value,
    )


@tf.function
def predict_step(model, inputs):
    # forward
    predictions = model(inputs=inputs, training=False)

    return predictions
