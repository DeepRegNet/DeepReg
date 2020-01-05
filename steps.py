import tensorflow as tf

import src.model.loss as loss


@tf.function
def train_step(model, optimizer, inputs, labels, fixed_grid_ref):
    # forward
    with tf.GradientTape() as tape:
        predictions = model(inputs=inputs, training=True)

        # loss
        loss_sim_value = loss.loss_similarity_fn(y_true=labels, y_pred=predictions)
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
def valid_step(model, inputs, labels, fixed_grid_ref):
    # forward
    predictions = model(inputs=inputs, training=False)

    # loss
    loss_sim_value = loss.loss_similarity_fn(y_true=labels, y_pred=predictions)
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
