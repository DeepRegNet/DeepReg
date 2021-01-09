import tensorflow as tf

from deepreg.model import layer, layer_util
from deepreg.registry import REGISTRY


def dict_without(d: dict, key) -> dict:
    copied = d.copy()
    copied.pop(key)
    return copied


class RegistrationModel(tf.keras.Model):
    def __init__(
        self,
        moving_image_size: tuple,
        fixed_image_size: tuple,
        index_size: int,
        labeled: bool,
        batch_size: int,
        config: dict,
    ):
        super().__init__()
        self.moving_image_size = moving_image_size
        self.fixed_image_size = fixed_image_size
        self.index_size = index_size
        self.labeled = labeled
        self.batch_size = batch_size
        self.config = config

        self._model = self.build_model()
        self.build_loss()

    def build_model(self):
        raise NotImplementedError

    def build_inputs(self):
        """
        Build input tensors.

        :return: tuple
        """
        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_image = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
        )
        # (batch, f_dim1, f_dim2, f_dim3, 1)
        fixed_image = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
        )
        # (batch, index_size)
        indices = tf.keras.Input(
            shape=(self.index_size,),
            batch_size=self.batch_size,
        )

        if not self.labeled:
            return moving_image, fixed_image, indices

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_label = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
        )
        # (batch, m_dim1, m_dim2, m_dim3, 1)
        fixed_label = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
        )
        return moving_image, fixed_image, indices, moving_label, fixed_label

    def concat_images(self, moving_image, fixed_image, moving_label=None):
        images = []

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_image = tf.expand_dims(moving_image, axis=4)
        moving_image = layer_util.resize3d(
            image=moving_image, size=self.fixed_image_size
        )
        images.append(moving_image)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        fixed_image = tf.expand_dims(fixed_image, axis=4)
        images.append(fixed_image)

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        if moving_label is not None:
            moving_label = tf.expand_dims(moving_label, axis=4)
            moving_label = layer_util.resize3d(
                image=moving_label, size=self.fixed_image_size
            )
            images.append(moving_label)

        # (batch, f_dim1, f_dim2, f_dim3, 2 or 3)
        images = tf.concat(images, axis=4)
        return images

    def _build_loss(self, name: str, inputs_dict):
        # build loss
        config = self.config["loss"][name]
        loss_cls = REGISTRY.build_loss(config=dict_without(d=config, key="weight"))
        loss = loss_cls(**inputs_dict)
        weighted_loss = loss * config["weight"]

        # add loss
        self._model.add_loss(weighted_loss)

        # add metric
        self._model.add_metric(
            loss, name=f"loss/{name}_{loss_cls.name}", aggregation="mean"
        )
        self._model.add_metric(
            weighted_loss,
            name=f"loss/{name}_{loss_cls.name}_weighted",
            aggregation="mean",
        )

    def build_loss(self):
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        return self._model(inputs, training=training, mask=mask)

    def postprocess(self, inputs, outputs):
        raise NotImplementedError


@REGISTRY.register_model(name="ddf")
class DDFModel(RegistrationModel):
    def build_model(self):
        # build inputs
        inputs = self.build_inputs()
        moving_image, fixed_image = inputs[:2]

        # build ddf
        backbone_inputs = self.concat_images(moving_image, fixed_image)
        backbone = REGISTRY.build_backbone(
            config=self.config["backbone"],
            default_args=dict(
                image_size=self.fixed_image_size,
                out_channels=3,
                out_kernel_initializer="zeros",
                out_activation=None,
            ),
        )
        # save backbone in case of affine to retrieve theta
        self._backbone = backbone

        # (f_dim1, f_dim2, f_dim3, 3)
        ddf = backbone(inputs=backbone_inputs)

        # build outputs
        warping = layer.Warping(fixed_image_size=self.fixed_image_size)
        # (f_dim1, f_dim2, f_dim3, 3)
        pred_fixed_image = warping(inputs=[ddf, moving_image])

        if not self.labeled:
            return tf.keras.Model(inputs=inputs, outputs=[ddf, pred_fixed_image])

        # (f_dim1, f_dim2, f_dim3, 3)
        moving_label = inputs[3]
        pred_fixed_label = warping(inputs=[ddf, moving_label])

        return tf.keras.Model(
            inputs=inputs, outputs=[ddf, pred_fixed_image, pred_fixed_label]
        )

    def build_loss(self):
        fixed_image = self._model.inputs[1]
        ddf, pred_fixed_image = self._model.outputs[:2]

        # ddf
        self._build_loss(name="regularization", inputs_dict=dict(inputs=ddf))

        # image
        self._build_loss(
            name="image", inputs_dict=dict(y_true=fixed_image, y_pred=pred_fixed_image)
        )

        # label
        if self.labeled:
            fixed_label = self._model.inputs[4]
            pred_fixed_label = self._model.outputs[2]
            self._build_loss(
                name="label",
                inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
            )

    def postprocess(self, inputs, outputs):
        moving_image, fixed_image, indices = inputs[:3]
        ddf, pred_fixed_image = outputs[:2]

        # each value is (tensor, normalize, on_label), where
        # - normalize = True if the tensor need to be normalized to [0, 1]
        # - on_label = True if the tensor depends on label
        processed = dict(
            moving_image=(moving_image, True, False),
            fixed_image=(fixed_image, True, False),
            ddf=(ddf, True, False),
            pred_fixed_image=(pred_fixed_image, True, False),
        )

        # save theta for affine model
        if hasattr(self._backbone, "theta"):
            processed["theta"] = (self._backbone.theta, None, None)

        if not self.labeled:
            return indices, processed

        moving_label, fixed_label = inputs[3:]
        pred_fixed_label = outputs[2]
        processed = {
            **dict(
                moving_label=(moving_label, False, True),
                fixed_label=(fixed_label, False, True),
                pred_fixed_label=(pred_fixed_label, False, True),
            ),
            **processed,
        }

        return indices, processed


@REGISTRY.register_model(name="dvf")
class DVFModel(RegistrationModel):
    def build_model(self):
        # build inputs
        inputs = self.build_inputs()
        moving_image, fixed_image = inputs[:2]

        # build ddf
        backbone_inputs = self.concat_images(moving_image, fixed_image)
        backbone = REGISTRY.build_backbone(
            config=self.config["backbone"],
            default_args=dict(
                image_size=self.fixed_image_size,
                out_channels=3,
                out_kernel_initializer="zeros",
                out_activation=None,
            ),
        )
        dvf = backbone(inputs=backbone_inputs)
        ddf = layer.IntDVF(fixed_image_size=self.fixed_image_size)(dvf)

        # build outputs
        warping = layer.Warping(fixed_image_size=self.fixed_image_size)
        # (f_dim1, f_dim2, f_dim3, 3)
        pred_fixed_image = warping(inputs=[ddf, moving_image])

        if not self.labeled:
            return tf.keras.Model(inputs=inputs, outputs=[dvf, ddf, pred_fixed_image])

        # (f_dim1, f_dim2, f_dim3, 3)
        moving_label = inputs[3]
        pred_fixed_label = warping(inputs=[ddf, moving_label])

        return tf.keras.Model(
            inputs=inputs, outputs=[dvf, ddf, pred_fixed_image, pred_fixed_label]
        )

    def build_loss(self):
        fixed_image = self._model.inputs[1]
        ddf, pred_fixed_image = self._model.outputs[1:3]

        # ddf
        self._build_loss(name="regularization", inputs_dict=dict(inputs=ddf))

        # image
        self._build_loss(
            name="image", inputs_dict=dict(y_true=fixed_image, y_pred=pred_fixed_image)
        )

        # label
        if self.labeled:
            fixed_label = self._model.inputs[4]
            pred_fixed_label = self._model.outputs[3]
            self._build_loss(
                name="label",
                inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
            )

    def postprocess(self, inputs, outputs):
        moving_image, fixed_image, indices = inputs[:3]
        dvf, ddf, pred_fixed_image = outputs[:3]

        # each value is (tensor, normalize, on_label), where
        # - normalize = True if the tensor need to be normalized to [0, 1]
        # - on_label = True if the tensor depends on label
        processed = dict(
            moving_image=(moving_image, True, False),
            fixed_image=(fixed_image, True, False),
            dvf=(dvf, True, False),
            ddf=(ddf, True, False),
            pred_fixed_image=(pred_fixed_image, True, False),
        )

        if not self.labeled:
            return indices, processed

        moving_label, fixed_label = inputs[3:]
        pred_fixed_label = outputs[3]
        processed = {
            **dict(
                moving_label=(moving_label, False, True),
                fixed_label=(fixed_label, False, True),
                pred_fixed_label=(pred_fixed_label, False, True),
            ),
            **processed,
        }

        return indices, processed


@REGISTRY.register_model(name="conditional")
class ConditionalModel(RegistrationModel):
    def build_model(self):
        assert self.labeled

        # build inputs
        inputs = self.build_inputs()
        moving_image, fixed_image = inputs[:2]
        moving_label = inputs[3]

        # build ddf
        backbone_inputs = self.concat_images(moving_image, fixed_image, moving_label)
        backbone = REGISTRY.build_backbone(
            config=self.config["backbone"],
            default_args=dict(
                image_size=self.fixed_image_size,
                out_channels=1,
                out_kernel_initializer="glorot_uniform",
                out_activation="sigmoid",
            ),
        )
        pred_fixed_label = backbone(inputs=backbone_inputs)

        return tf.keras.Model(inputs=inputs, outputs=pred_fixed_label)

    def build_loss(self):
        fixed_label = self._model.inputs[4]
        pred_fixed_label = self._model.outputs[0]
        self._build_loss(
            name="label",
            inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
        )

    def postprocess(self, inputs, outputs):
        moving_image, fixed_image, indices, moving_label, fixed_label = inputs
        pred_fixed_image = outputs[0]

        # each value is (tensor, normalize, on_label), where
        # - normalize = True if the tensor need to be normalized to [0, 1]
        # - on_label = True if the tensor depends on label
        processed = dict(
            moving_image=(moving_image, True, False),
            fixed_image=(fixed_image, True, False),
            pred_fixed_image=(pred_fixed_image, True, True),
            moving_label=(moving_label, False, True),
            fixed_label=(fixed_label, False, True),
        )

        return indices, processed
