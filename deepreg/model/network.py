'''
This module parses the netwotk configurations to build network backbone architecture (including conditional segmentation), inputs, DDF/DVF outputs and loss
'''

import tensorflow as tf

import deepreg.model.layer as layer
import deepreg.model.loss.deform
import deepreg.model.loss.image as image_loss
import deepreg.model.loss.label as label_loss
from deepreg.model.backbone.local_net import LocalNet
from deepreg.model.backbone.u_net import UNet


def build_backbone(image_size, out_channels, tf_model_config):
    """
    backbone model accepts a single input of shape [batch, dim1, dim2, dim3, ch_in]
               and returns a single output of shape [batch, dim1, dim2, dim3, ch_out]

    :param image_size: [dim1, dim2, dim3]
    :param out_channels: ch_out
    :param tf_model_config: model configuration, e.g. key-value pairs return from parser.yaml.load
    :return: tf.keras.Model object
    """

    # no activation
    if tf_model_config["backbone"]["out_activation"] == "":
        tf_model_config["backbone"]["out_activation"] = None

    if tf_model_config["backbone"]["name"] == "local":
        return LocalNet(image_size=image_size, out_channels=out_channels,
                        out_kernel_initializer=tf_model_config["backbone"]["out_kernel_initializer"],
                        out_activation=tf_model_config["backbone"]["out_activation"],
                        **tf_model_config["local"])
    elif tf_model_config["backbone"]["name"] == "unet":
        return UNet(image_size=image_size, out_channels=out_channels,
                    out_kernel_initializer=tf_model_config["backbone"]["out_kernel_initializer"],
                    out_activation=tf_model_config["backbone"]["out_activation"],
                    **tf_model_config["unet"])
    else:
        raise ValueError("Unknown model name")


def build_inputs(moving_image_size, fixed_image_size, index_size, batch_size):
    """
    Configure a pair of moving and fixed images and a pair of moving and fixed labels as model input 
               and returns model input tf.keras.Input
    
    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param batch_size: minibatch size 
    :return: tf.keras.Input object
    """
    moving_image = tf.keras.Input(shape=(*moving_image_size,), batch_size=batch_size,
                                  name="moving_image")  # [batch, m_dim1, m_dim2, m_dim3]
    fixed_image = tf.keras.Input(shape=(*fixed_image_size,), batch_size=batch_size,
                                 name="fixed_image")  # [batch, f_dim1, f_dim2, f_dim3]
    moving_label = tf.keras.Input(shape=(*moving_image_size,), batch_size=batch_size,
                                  name="moving_label")  # [batch, m_dim1, m_dim2, m_dim3]
    indices = tf.keras.Input(shape=(index_size,), batch_size=batch_size,
                             name="indices")  # [batch, 2]
    return moving_image, fixed_image, moving_label, indices


def build_ddf_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config, tf_loss_config):
    """
    Build the model if the output is DDF (dense displacement field)

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param batch_size: minibatch size
    :param tf_model_config: model configuration, e.g. key-value pairs return from parser.yaml.load
    :param tf_loss_config: loss configuration, e.g. key-value pairs return from parser.yaml.load
    :return: the built tf.keras.Model
    """

    def forward(_backbone, _moving_image, _moving_label, _fixed_image):
        """
        Perform the network forward pass

        :param _backbone: model architecture object, e.g. return from model.backbone.local_net
        :param _moving_image: [batch, m_dim1, m_dim2, m_dim3]
        :param _moving_label: [batch, m_dim1, m_dim2, m_dim3]
        :param _fixed_image:  [batch, f_dim1, f_dim2, f_dim3]
        :return: tuple(_ddf, _pred_fixed_image, _pred_fixed_label)
            WHEHE
            str _ddf is the dense displacement field [batch, m_dim1, m_dim2, m_dim3, 3]
            str _pred_fixed_image is the predicted (warped) moving image [batch, f_dim1, f_dim2, f_dim3]
            str _pred_fixed_label is the predicted (warped) moving label [batch, f_dim1, f_dim2, f_dim3]
        """
        # ddf
        backbone_input = tf.concat([layer.Resize3d(size=fixed_image_size)(inputs=tf.expand_dims(_moving_image, axis=4)),
                                    tf.expand_dims(_fixed_image, axis=4)],
                                   axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 2]
        _ddf = _backbone(inputs=backbone_input)  # [batch, f_dim1, f_dim2, f_dim3, 3]

        # prediction image ang label shape = [batch, f_dim1, f_dim2, f_dim3]
        _pred_fixed_image = layer.Warping(fixed_image_size=fixed_image_size)([_ddf, _moving_image])
        _pred_fixed_label = layer.Warping(fixed_image_size=fixed_image_size)([_ddf, _moving_label])

        return _ddf, _pred_fixed_image, _pred_fixed_label

    def add_loss_metric(_fixed_image, _pred_fixed_image, _ddf, _fixed_label, _pred_fixed_label, suffix):
        """
        Configue and add the training loss, including image, label and deformation regularisation

        :param _fixed_image:      [batch, f_dim1, f_dim2, f_dim3]
        :param _pred_fixed_image: [batch, f_dim1, f_dim2, f_dim3]
        :param _ddf:              [batch, f_dim1, f_dim2, f_dim3, 3]
        :param _fixed_label:      [batch, f_dim1, f_dim2, f_dim3]
        :param _pred_fixed_label: [batch, f_dim1, f_dim2, f_dim3]
        :param suffix: string reserved or extra information 
        :return: tf.keras.Model with loss and metric added
        """
        # image loss
        if tf_loss_config["similarity"]["image"]["weight"] > 0:
            loss_image = tf.reduce_mean(image_loss.similarity_fn(
                y_true=_fixed_image, y_pred=_pred_fixed_image,
                **tf_loss_config["similarity"]["image"]))
            weighted_loss_image = loss_image * tf_loss_config["similarity"]["image"]["weight"]
            model.add_loss(weighted_loss_image)
            model.add_metric(loss_image, name="loss/image_similarity" + suffix, aggregation="mean")
            model.add_metric(weighted_loss_image, name="loss/weighted_image_similarity" + suffix, aggregation="mean")

        # regularization loss
        loss_reg = tf.reduce_mean(
            deepreg.model.loss.deform.local_displacement_energy(_ddf, **tf_loss_config["regularization"]))
        weighted_loss_reg = loss_reg * tf_loss_config["regularization"]["weight"]
        model.add_loss(weighted_loss_reg)
        model.add_metric(loss_reg, name="loss/regularization" + suffix, aggregation="mean")
        model.add_metric(weighted_loss_reg, name="loss/weighted_regularization" + suffix, aggregation="mean")

        # label loss
        if _fixed_label is not None:
            label_loss_fn = label_loss.get_similarity_fn(config=tf_loss_config["similarity"]["label"])
            loss_label = label_loss_fn(y_true=_fixed_label, y_pred=_pred_fixed_label)
            model.add_loss(loss_label)
            model.add_metric(loss_label, name="loss/label" + suffix, aggregation="mean")

    # inputs
    moving_image, fixed_image, moving_label, indices = build_inputs(
        moving_image_size, fixed_image_size, index_size, batch_size)

    # backbone
    backbone = build_backbone(image_size=fixed_image_size, out_channels=3,
                              tf_model_config=tf_model_config)

    # forward
    ddf, pred_fixed_image, pred_fixed_label = forward(_backbone=backbone,
                                                      _moving_image=moving_image,
                                                      _moving_label=moving_label,
                                                      _fixed_image=fixed_image)

    # build model
    model = tf.keras.Model(inputs=[moving_image, fixed_image, moving_label, indices],
                           outputs=[pred_fixed_label],
                           name="DDFRegModel")
    model.ddf = ddf

    # loss and metric
    add_loss_metric(_fixed_image=fixed_image,
                    _pred_fixed_image=pred_fixed_image,
                    _ddf=ddf,
                    _fixed_label=None,
                    _pred_fixed_label=None,
                    suffix="")

    return model


def build_dvf_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config, tf_loss_config):
    """
    Build the model if the output is DVF-integrated DDF (dense displacement field)

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param batch_size: minibatch size
    :param tf_model_config: model configuration, e.g. key-value pairs return from parser.yaml.load
    :param tf_loss_config: loss configuration, e.g. key-value pairs return from parser.yaml.load
    :return: the built tf.keras.Model
    """
    

    def forward(_backbone, _moving_image, _moving_label, _fixed_image):
        """
        Perform the network forward pass

        :param _backbone: model architecture object, e.g. return from model.backbone.local_net
        :param _moving_image: [batch, m_dim1, m_dim2, m_dim3]
        :param _moving_label: [batch, m_dim1, m_dim2, m_dim3]
        :param _fixed_image:  [batch, f_dim1, f_dim2, f_dim3]
        :return: tuple(_dvf, _ddf, _pred_fixed_image, _pred_fixed_label)
            WHEHE
            str _dvf is the dense velocity field [batch, m_dim1, m_dim2, m_dim3, 3]
            str _ddf is the dense displacement field [batch, m_dim1, m_dim2, m_dim3, 3]
            str _pred_fixed_image is the predicted (warped) moving image [batch, f_dim1, f_dim2, f_dim3]
            str _pred_fixed_label is the predicted (warped) moving label [batch, f_dim1, f_dim2, f_dim3]
        """
        # ddf
        backbone_input = tf.concat([layer.Resize3d(size=fixed_image_size)(inputs=tf.expand_dims(_moving_image, axis=4)),
                                    tf.expand_dims(_fixed_image, axis=4)],
                                   axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 2]
        _dvf = _backbone(inputs=backbone_input)  # [batch, f_dim1, f_dim2, f_dim3, 3]
        _ddf = layer.IntDVF(fixed_image_size=fixed_image_size)(_dvf)

        # prediction image ang label shape = [batch, f_dim1, f_dim2, f_dim3]
        _pred_fixed_image = layer.Warping(fixed_image_size=fixed_image_size)([_ddf, _moving_image])
        _pred_fixed_label = layer.Warping(fixed_image_size=fixed_image_size)([_ddf, _moving_label])

        return _dvf, _ddf, _pred_fixed_image, _pred_fixed_label

    def add_loss_metric(_fixed_image, _pred_fixed_image, _ddf, _fixed_label, _pred_fixed_label, suffix):
        """
        Configue and add the training loss, including image, label and deformation regularisation

        :param _fixed_image:      [batch, f_dim1, f_dim2, f_dim3]
        :param _pred_fixed_image: [batch, f_dim1, f_dim2, f_dim3]
        :param _ddf:              [batch, f_dim1, f_dim2, f_dim3, 3]
        :param _fixed_label:      [batch, f_dim1, f_dim2, f_dim3]
        :param _pred_fixed_label: [batch, f_dim1, f_dim2, f_dim3]
        :param suffix: string reserved or extra information 
        :return: tf.keras.Model with loss and metric added
        """
        # image loss
        if tf_loss_config["similarity"]["image"]["weight"] > 0:
            loss_image = tf.reduce_mean(image_loss.similarity_fn(
                y_true=_fixed_image, y_pred=_pred_fixed_image,
                **tf_loss_config["similarity"]["image"]))
            weighted_loss_image = loss_image * tf_loss_config["similarity"]["image"]["weight"]
            model.add_loss(weighted_loss_image)
            model.add_metric(loss_image, name="loss/image_similarity" + suffix, aggregation="mean")
            model.add_metric(weighted_loss_image, name="loss/weighted_image_similarity" + suffix, aggregation="mean")

        # regularization loss
        loss_reg = tf.reduce_mean(
            deepreg.model.loss.deform.local_displacement_energy(_ddf, **tf_loss_config["regularization"]))
        weighted_loss_reg = loss_reg * tf_loss_config["regularization"]["weight"]
        model.add_loss(weighted_loss_reg)
        model.add_metric(loss_reg, name="loss/regularization" + suffix, aggregation="mean")
        model.add_metric(weighted_loss_reg, name="loss/weighted_regularization" + suffix, aggregation="mean")

        # label loss
        if _fixed_label is not None:
            label_loss_fn = label_loss.get_similarity_fn(config=tf_loss_config["similarity"]["label"])
            loss_label = label_loss_fn(y_true=_fixed_label, y_pred=_pred_fixed_label)
            model.add_loss(loss_label)
            model.add_metric(loss_label, name="loss/label" + suffix, aggregation="mean")

    # inputs
    moving_image, fixed_image, moving_label, indices = build_inputs(
        moving_image_size, fixed_image_size, index_size, batch_size)

    # backbone
    backbone = build_backbone(image_size=fixed_image_size, out_channels=3,
                              tf_model_config=tf_model_config)

    # forward
    dvf, ddf, pred_fixed_image, pred_fixed_label = forward(_backbone=backbone,
                                                           _moving_image=moving_image,
                                                           _moving_label=moving_label,
                                                           _fixed_image=fixed_image)

    # build model
    model = tf.keras.Model(inputs=[moving_image, fixed_image, moving_label, indices],
                           outputs=[pred_fixed_label],
                           name="DDFRegModel")
    model.dvf = dvf
    model.ddf = ddf

    # loss and metric
    add_loss_metric(_fixed_image=fixed_image,
                    _pred_fixed_image=pred_fixed_image,
                    _ddf=ddf,
                    _fixed_label=None,
                    _pred_fixed_label=None,
                    suffix="")

    return model


def build_cond_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config, tf_loss_config):
    """
    Build the model if using conditional segmentation algorithm

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param batch_size: minibatch size
    :param tf_model_config: model configuration, e.g. key-value pairs return from parser.yaml.load
    :param tf_loss_config: loss configuration, e.g. key-value pairs return from parser.yaml.load
    :return: the built tf.keras.Model
    """
    # inputs
    moving_image, fixed_image, moving_label, indices = build_inputs(
        moving_image_size, fixed_image_size, index_size, batch_size)
    backbone_input = tf.concat([layer.Resize3d(size=fixed_image_size)(inputs=tf.expand_dims(moving_image, axis=4)),
                                tf.expand_dims(fixed_image, axis=4),
                                layer.Resize3d(size=fixed_image_size)(inputs=tf.expand_dims(moving_label, axis=4)),
                                ],
                               axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 3]

    # backbone
    backbone = build_backbone(image_size=fixed_image_size, out_channels=1,
                              tf_model_config=tf_model_config)

    # prediction
    pred_fixed_label = backbone(inputs=backbone_input)  # [batch, f_dim1, f_dim2, f_dim3, 1]
    pred_fixed_label = tf.squeeze(pred_fixed_label, axis=4)

    # build model
    model = tf.keras.Model(inputs=[moving_image, fixed_image, moving_label, indices],
                           outputs=[pred_fixed_label],
                           name="CondRegModel")

    return model


def build_seg_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config, tf_loss_config):
    """
    @Experimental
    Build the model for a segmenation model for the fixed image

    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param batch_size: minibatch size
    :param tf_model_config: model configuration, e.g. key-value pairs return from parser.yaml.load
    :param tf_loss_config: loss configuration, e.g. key-value pairs return from parser.yaml.load
    :return: the built tf.keras.Model
    """
    # inputs
    moving_image, fixed_image, moving_label, indices = build_inputs(
        moving_image_size, fixed_image_size, index_size, batch_size)
    backbone_input = tf.expand_dims(fixed_image, axis=4)  # [batch, f_dim1, f_dim2, f_dim3, 1]

    # backbone
    backbone = build_backbone(image_size=fixed_image_size, out_channels=1,
                              tf_model_config=tf_model_config)

    # prediction
    pred_fixed_label = backbone(inputs=backbone_input)  # [batch, f_dim1, f_dim2, f_dim3, 1]
    pred_fixed_label = tf.squeeze(pred_fixed_label, axis=4)

    # build model
    model = tf.keras.Model(inputs=[moving_image, fixed_image, moving_label, indices],
                           outputs=[pred_fixed_label],
                           name="SegModel")

    return model


def build_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config, tf_loss_config):    
    """
    Parsing algorithm types to model building functions
    
    :param moving_image_size: [m_dim1, m_dim2, m_dim3]
    :param fixed_image_size: [f_dim1, f_dim2, f_dim3]
    :param index_size: dataset size
    :param batch_size: minibatch size 
    :param tf_model_config: model configuration, e.g. key-value pairs return from parser.yaml.load
    :param tf_loss_config: loss configuration, e.g. key-value pairs return from parser.yaml.load
    :return: the built tf.keras.Model
    """

    if tf_model_config["method"] == "ddf":
        return build_ddf_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config,
                               tf_loss_config)
    elif tf_model_config["method"] == "dvf":
        return build_dvf_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config,
                               tf_loss_config)
    elif tf_model_config["method"] == "conditional":
        return build_cond_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config,
                                tf_loss_config)
    elif tf_model_config["method"] == "seg":
        return build_seg_model(moving_image_size, fixed_image_size, index_size, batch_size, tf_model_config,
                               tf_loss_config)
    else:
        raise ValueError("Unknown model method")
