import src.model.loss as loss
import tensorflow as tf

import deepreg.model.loss.image as image_loss
import deepreg.model.loss.label as label_loss


@tf.function
def train_step(args_dict, model, optimizer, inputs, labels, fixed_grid_ref):  # modified
    # forward
    with tf.GradientTape() as tape:
        warped_moving_image, warped_moving_label, hm_0 = model(
            inputs=inputs, training=True
        )
        moving_image = tf.expand_dims(inputs[0], axis=4)
        fixed_image = tf.expand_dims(inputs[1], axis=4)
        fixed_label = labels
        warped_moving_label = tf.squeeze(warped_moving_label)

        # loss
        if args_dict["loss_type"] == "ncc":
            loss_sim_value = (
                -image_loss.local_normalized_cross_correlation(
                    y_true=fixed_image, y_pred=warped_moving_image
                )
                * args_dict["w_ncc"]
            )
        elif args_dict["loss_type"] == "ssd":
            loss_sim_value = (
                image_loss.ssd(y_true=fixed_image, y_pred=warped_moving_image)
                * args_dict["w_ssd"]
            )
        else:
            print("loss type wrong")
            raise NotImplementedError

        x1 = tf.gather(
            hm_0, indices=[i for i in range(args_dict["batch_size"]) if (i % 2) == 0]
        )  # [n, 8, 8, 7, 256]
        x2 = tf.gather(
            hm_0, indices=[i for i in range(args_dict["batch_size"]) if (i % 2) != 0]
        )
        loss_mmd_value = loss.loss_mmd(x1, x2, args_dict["sigmas"]) * args_dict["w_mmd"]

        loss_reg_value = sum(model.losses) * args_dict["w_bde"]
        loss_dice_value = (
            label_loss.single_scale_loss(fixed_label, warped_moving_label, "dice")
            * args_dict["w_dce"]
        )
        loss_total_value = (
            loss_sim_value + loss_reg_value + loss_dice_value + loss_mmd_value
        )

        # metrics
        metric_dice_value = label_loss.dice_score(
            fixed_label, warped_moving_label, binary=True
        )
        metric_dist_value = label_loss.compute_centroid_distance(
            fixed_label, warped_moving_label, fixed_grid_ref
        )

    # optimize
    grads = tape.gradient(loss_total_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    # opt params
    opt_lr_value = optimizer._decayed_lr("float32")

    return dict(
        loss_sim=loss_sim_value,
        loss_reg=loss_reg_value,
        loss_mmd=loss_mmd_value,
        loss_dice=loss_dice_value,
        loss_total=loss_total_value,
        metric_dice=metric_dice_value,
        metric_dist=metric_dist_value,
        opt_lr=opt_lr_value,
    )


@tf.function
def valid_step(args_dict, model, inputs, labels, fixed_grid_ref, return_type="metrics"):
    # forward
    moving_image, fixed_image, fixed_label = inputs[0], inputs[1], labels
    if args_dict["test_before_reg"]:
        warped_moving_image, warped_moving_label = (
            tf.expand_dims(inputs[0], axis=4),
            inputs[2],
        )
    else:
        warped_moving_image, warped_moving_label, hm_0 = model(
            inputs=inputs, training=False
        )
    moving_image = tf.expand_dims(inputs[0], axis=4)
    fixed_image = tf.expand_dims(inputs[1], axis=4)
    fixed_label = labels
    warped_moving_label = tf.squeeze(warped_moving_label)
    # loss
    if args_dict["loss_type"] == "ncc":
        loss_sim_value = (
            -image_loss.local_normalized_cross_correlation(
                y_true=fixed_image, y_pred=warped_moving_image
            )
            * args_dict["w_ncc"]
        )
    elif args_dict["loss_type"] == "ssd":
        loss_sim_value = (
            image_loss.ssd(y_true=fixed_image, y_pred=warped_moving_image)
            * args_dict["w_ssd"]
        )
    else:
        print("loss type wrong")
        raise NotImplementedError

    if args_dict["test_before_reg"]:
        loss_reg_value = 0
    else:
        loss_reg_value = sum(model.losses) * args_dict["w_bde"]

    if len(warped_moving_label.shape) == 3:
        warped_moving_label = warped_moving_label[None, ...]

    loss_dice_value = (
        label_loss.single_scale_loss(fixed_label, warped_moving_label, "dice")
        * args_dict["w_dce"]
    )
    loss_total_value = loss_sim_value + loss_reg_value + loss_dice_value

    if not args_dict["test_mode"]:
        # when testing, batchsize is 1
        print("not in test mode...")
        x1 = tf.gather(
            hm_0, indices=[i for i in range(args_dict["batch_size"]) if (i % 2) == 0]
        )  # [n, 8, 8, 7, 256]
        x2 = tf.gather(
            hm_0, indices=[i for i in range(args_dict["batch_size"]) if (i % 2) != 0]
        )
        loss_mmd_value = loss.loss_mmd(x1, x2, args_dict["sigmas"]) * args_dict["w_mmd"]
        loss_total_value += loss_mmd_value

    # metrics
    metric_dice_value = label_loss.dice_score(
        fixed_label, warped_moving_label, binary=True
    )
    metric_dist_value = label_loss.compute_centroid_distance(
        fixed_label, warped_moving_label, fixed_grid_ref
    )

    if return_type == "metrics":
        return dict(
            loss_sim=loss_sim_value,
            loss_reg=loss_reg_value,
            loss_mmd=loss_mmd_value,
            loss_dice=loss_dice_value,
            loss_total=loss_total_value,
            metric_dice=metric_dice_value,
            metric_dist=metric_dist_value,
        )
    elif return_type == "prediction":
        return (
            metric_dice_value,
            metric_dist_value,
            loss_sim_value,
            [warped_moving_image, warped_moving_label],
        )
    else:
        pass
