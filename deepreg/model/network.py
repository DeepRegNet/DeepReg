import logging
from abc import abstractmethod
from copy import deepcopy
from typing import Dict, Optional

import tensorflow as tf

from deepreg.model import layer, layer_util
from deepreg.model.backbone import GlobalNet
from deepreg.registry import REGISTRY


def dict_without(d: dict, key) -> dict:
    """
    Return a copy of the given dict without a certain key.

    :param d: dict to be copied.
    :param key: key to be removed.
    :return: the copy without a key
    """
    copied = deepcopy(d)
    copied.pop(key)
    return copied


class RegistrationModel(tf.keras.Model):
    """Interface for registration model."""

    def __init__(
        self,
        moving_image_size: tuple,
        fixed_image_size: tuple,
        index_size: int,
        labeled: bool,
        batch_size: int,
        config: dict,
        num_devices: int = 1,
        name: str = "RegistrationModel",
    ):
        """
        Init.

        :param moving_image_size: (m_dim1, m_dim2, m_dim3)
        :param fixed_image_size: (f_dim1, f_dim2, f_dim3)
        :param index_size: number of indices for identify each sample
        :param labeled: if the data is labeled
        :param batch_size: size of mini-batch
        :param config: config for method, backbone, and loss.
        :param num_devices: number of GPU used,
            global_batch_size = batch_size*num_devices
        :param name: name of the model
        """
        super().__init__(name=name)
        self.moving_image_size = moving_image_size
        self.fixed_image_size = fixed_image_size
        self.index_size = index_size
        self.labeled = labeled
        self.batch_size = batch_size
        self.config = config
        self.num_devices = num_devices
        self.global_batch_size = num_devices * batch_size

        self._inputs = None  # save inputs of self._model as dict
        self._outputs = None  # save outputs of self._model as dict
        self._model = self.build_model()
        self.build_loss()

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        return dict(
            moving_image_size=self.moving_image_size,
            fixed_image_size=self.fixed_image_size,
            index_size=self.index_size,
            labeled=self.labeled,
            batch_size=self.batch_size,
            config=self.config,
            num_devices=self.num_devices,
            name=self.name,
        )

    @abstractmethod
    def build_model(self):
        """Build the model to be saved as self._model."""

    def build_inputs(self) -> Dict[str, tf.keras.layers.Input]:
        """
        Build input tensors.

        :return: dict of inputs.
        """
        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_image = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
            name="moving_image",
        )
        # (batch, f_dim1, f_dim2, f_dim3, 1)
        fixed_image = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
            name="fixed_image",
        )
        # (batch, index_size)
        indices = tf.keras.Input(
            shape=(self.index_size,),
            batch_size=self.batch_size,
            name="indices",
        )

        if not self.labeled:
            return dict(
                moving_image=moving_image, fixed_image=fixed_image, indices=indices
            )

        # (batch, m_dim1, m_dim2, m_dim3, 1)
        moving_label = tf.keras.Input(
            shape=self.moving_image_size,
            batch_size=self.batch_size,
            name="moving_label",
        )
        # (batch, m_dim1, m_dim2, m_dim3, 1)
        fixed_label = tf.keras.Input(
            shape=self.fixed_image_size,
            batch_size=self.batch_size,
            name="fixed_label",
        )
        return dict(
            moving_image=moving_image,
            fixed_image=fixed_image,
            moving_label=moving_label,
            fixed_label=fixed_label,
            indices=indices,
        )

    def concat_images(
        self,
        moving_image: tf.Tensor,
        fixed_image: tf.Tensor,
        moving_label: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """
        Adjust image shape and concatenate them together.

        :param moving_image: registration source
        :param fixed_image: registration target
        :param moving_label: optional, only used for conditional model.
        :return:
        """
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

    def _build_loss(self, name: str, inputs_dict: dict):
        """
        Build and add one weighted loss together with the metrics.

        :param name: name of loss
        :param inputs_dict: inputs for loss function
        """
        if name not in self.config["loss"]:
            # loss config is not defined
            logging.warning(
                f"The configuration for loss {name} is not defined."
                f"Loss is not used."
            )
            return

        loss_config = self.config["loss"][name]

        if "weight" not in loss_config:
            # default loss weight 1
            logging.warning(
                f"The weight for loss {name} is not defined."
                f"Default weight = 1.0 is used."
            )
            loss_config["weight"] = 1.0

        # build loss
        weight = loss_config["weight"]

        if weight == 0:
            logging.warning(f"The weight for loss {name} is zero." f"Loss is not used.")
            return

        loss_cls = REGISTRY.build_loss(config=dict_without(d=loss_config, key="weight"))
        loss = loss_cls(**inputs_dict) / self.global_batch_size
        weighted_loss = loss * weight

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

    @abstractmethod
    def build_loss(self):
        """Build losses according to configs."""

    def call(
        self, inputs: Dict[str, tf.Tensor], training=None, mask=None
    ) -> Dict[str, tf.Tensor]:
        """
        Call the self._model.

        :param inputs: a dict of tensors.
        :param training: training or not.
        :param mask: maks for inputs.
        :return:
        """
        return self._model(inputs, training=training, mask=mask)  # pragma: no cover

    @abstractmethod
    def postprocess(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> (tf.Tensor, Dict):
        """
        Return a dict used for saving inputs and outputs.

        :param inputs: dict of model inputs
        :param outputs: dict of model outputs
        :return: tuple, indices and a dict.
            In the dict, each value is (tensor, normalize, on_label), where
            - normalize = True if the tensor need to be normalized to [0, 1]
            - on_label = True if the tensor depends on label
        """


@REGISTRY.register_model(name="ddf")
class DDFModel(RegistrationModel):
    """
    A registration model predicts DDF.

    When using global net as backbone,
    the model predicts an affine transformation parameters,
    and a DDF is calculated based on that.
    """

    def build_model(self):
        """Build the model to be saved as self._model."""
        # build inputs
        self._inputs = self.build_inputs()
        moving_image = self._inputs["moving_image"]
        fixed_image = self._inputs["fixed_image"]

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

        if isinstance(backbone, GlobalNet):
            # (f_dim1, f_dim2, f_dim3, 3), (4, 3)
            ddf, theta = backbone(inputs=backbone_inputs)
            self._outputs = dict(ddf=ddf, theta=theta)
        else:
            # (f_dim1, f_dim2, f_dim3, 3)
            ddf = backbone(inputs=backbone_inputs)
            self._outputs = dict(ddf=ddf)

        # build outputs
        warping = layer.Warping(fixed_image_size=self.fixed_image_size)
        # (f_dim1, f_dim2, f_dim3, 3)
        pred_fixed_image = warping(inputs=[ddf, moving_image])
        self._outputs["pred_fixed_image"] = pred_fixed_image

        if not self.labeled:
            return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

        # (f_dim1, f_dim2, f_dim3, 3)
        moving_label = self._inputs["moving_label"]
        pred_fixed_label = warping(inputs=[ddf, moving_label])

        self._outputs["pred_fixed_label"] = pred_fixed_label
        return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

    def build_loss(self):
        """Build losses according to configs."""
        fixed_image = self._inputs["fixed_image"]
        ddf = self._outputs["ddf"]
        pred_fixed_image = self._outputs["pred_fixed_image"]

        # ddf
        self._build_loss(name="regularization", inputs_dict=dict(inputs=ddf))

        # image
        self._build_loss(
            name="image", inputs_dict=dict(y_true=fixed_image, y_pred=pred_fixed_image)
        )

        # label
        if self.labeled:
            fixed_label = self._inputs["fixed_label"]
            pred_fixed_label = self._outputs["pred_fixed_label"]
            self._build_loss(
                name="label",
                inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
            )

    def postprocess(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> (tf.Tensor, Dict):
        """
        Return a dict used for saving inputs and outputs.

        :param inputs: dict of model inputs
        :param outputs: dict of model outputs
        :return: tuple, indices and a dict.
            In the dict, each value is (tensor, normalize, on_label), where
            - normalize = True if the tensor need to be normalized to [0, 1]
            - on_label = True if the tensor depends on label
        """
        indices = inputs["indices"]
        processed = dict(
            moving_image=(inputs["moving_image"], True, False),
            fixed_image=(inputs["fixed_image"], True, False),
            ddf=(outputs["ddf"], True, False),
            pred_fixed_image=(outputs["pred_fixed_image"], True, False),
        )

        # save theta for affine model
        if "theta" in outputs:
            processed["theta"] = (outputs["theta"], None, None)

        if not self.labeled:
            return indices, processed

        processed = {
            **dict(
                moving_label=(inputs["moving_label"], False, True),
                fixed_label=(inputs["fixed_label"], False, True),
                pred_fixed_label=(outputs["pred_fixed_label"], False, True),
            ),
            **processed,
        }

        return indices, processed


@REGISTRY.register_model(name="dvf")
class DVFModel(DDFModel):
    """
    A registration model predicts DVF.

    DDF is calculated based on DVF.
    """

    def build_model(self):
        """Build the model to be saved as self._model."""
        # build inputs
        self._inputs = self.build_inputs()
        moving_image = self._inputs["moving_image"]
        fixed_image = self._inputs["fixed_image"]

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

        self._outputs = dict(dvf=dvf, ddf=ddf, pred_fixed_image=pred_fixed_image)

        if not self.labeled:
            return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

        # (f_dim1, f_dim2, f_dim3, 3)
        moving_label = self._inputs["moving_label"]
        pred_fixed_label = warping(inputs=[ddf, moving_label])

        self._outputs["pred_fixed_label"] = pred_fixed_label
        return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

    def postprocess(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> (tf.Tensor, Dict):
        """
        Return a dict used for saving inputs and outputs.

        :param inputs: dict of model inputs
        :param outputs: dict of model outputs
        :return: tuple, indices and a dict.
            In the dict, each value is (tensor, normalize, on_label), where
            - normalize = True if the tensor need to be normalized to [0, 1]
            - on_label = True if the tensor depends on label
        """
        indices, processed = super().postprocess(inputs=inputs, outputs=outputs)
        processed["dvf"] = (outputs["dvf"], True, False)
        return indices, processed


@REGISTRY.register_model(name="conditional")
class ConditionalModel(RegistrationModel):
    """
    A registration model predicts fixed image label without DDF or DVF.
    """

    def build_model(self):
        """Build the model to be saved as self._model."""
        assert self.labeled

        # build inputs
        self._inputs = self.build_inputs()
        moving_image = self._inputs["moving_image"]
        fixed_image = self._inputs["fixed_image"]
        moving_label = self._inputs["moving_label"]

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
        # (batch, f_dim1, f_dim2, f_dim3)
        pred_fixed_label = backbone(inputs=backbone_inputs)
        pred_fixed_label = tf.squeeze(pred_fixed_label, axis=4)

        self._outputs = dict(pred_fixed_label=pred_fixed_label)
        return tf.keras.Model(inputs=self._inputs, outputs=self._outputs)

    def build_loss(self):
        """Build losses according to configs."""
        fixed_label = self._inputs["fixed_label"]
        pred_fixed_label = self._outputs["pred_fixed_label"]

        self._build_loss(
            name="label",
            inputs_dict=dict(y_true=fixed_label, y_pred=pred_fixed_label),
        )

    def postprocess(
        self,
        inputs: Dict[str, tf.Tensor],
        outputs: Dict[str, tf.Tensor],
    ) -> (tf.Tensor, Dict):
        """
        Return a dict used for saving inputs and outputs.

        :param inputs: dict of model inputs
        :param outputs: dict of model outputs
        :return: tuple, indices and a dict.
            In the dict, each value is (tensor, normalize, on_label), where
            - normalize = True if the tensor need to be normalized to [0, 1]
            - on_label = True if the tensor depends on label
        """
        indices = inputs["indices"]
        processed = dict(
            moving_image=(inputs["moving_image"], True, False),
            fixed_image=(inputs["fixed_image"], True, False),
            pred_fixed_label=(outputs["pred_fixed_label"], True, True),
            moving_label=(inputs["moving_label"], False, True),
            fixed_label=(inputs["fixed_label"], False, True),
        )

        return indices, processed
