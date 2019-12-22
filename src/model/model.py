import tensorflow as tf


def model_fn(features, labels, mode, params):
    MODEL = {'features': features, 'labels': labels, 'mode': mode, 'params': params}

    # send the features through the graph
    MODEL = build_fn(MODEL)

    # prediction
    MODEL['predictions'] = {'labels': MODEL['net_logits']}

    MODEL['export_outputs'] = {
        k: tf.estimator.export.PredictOutput(v) for k, v in MODEL['predictions'].items()
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return mode_predict(MODEL)

    # calculate the loss
    MODEL = loss_fn(MODEL)

    # calculate all metrics and send them to tf.summary
    MODEL = metrics_fn(MODEL)

    if mode == tf.estimator.ModeKeys.EVAL:
        return mode_eval(MODEL)

    if mode == tf.estimator.ModeKeys.TRAIN:
        return mode_train(MODEL)


def mode_train(model):
    # extract variables for easier reading here
    global_step = tf.train.get_global_step()
    learning_rate = model['params']['learning_rate']
    loss = model['loss']

    # do the training here
    model['optimizer'] = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    model['train_op'] = model['optimizer'].minimize(loss, global_step=global_step)

    spec = tf.estimator.EstimatorSpec(
        mode=model['mode'],
        loss=model['loss'],
        train_op=model['train_op'],
        eval_metric_ops=model['metrics'],
        predictions=model['predictions'],
        export_outputs=model['export_outputs']
    )
    return spec
