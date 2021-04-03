"""Provide different loss or metrics classes for labels."""

from typing import List, Optional

import tensorflow as tf

from deepreg.constant import EPS
from deepreg.loss.util import NegativeLossMixin, cauchy_kernel1d
from deepreg.loss.util import gaussian_kernel1d_sigma as gaussian_kernel1d
from deepreg.loss.util import separable_filter
from deepreg.registry import REGISTRY


class MultiScaleLoss(tf.keras.losses.Loss):
    """
    Base class for multi-scale loss.

    It applies the loss at different scales (gaussian or cauchy smoothing).
    It is assumed that loss values are between 0 and 1.
    """

    kernel_fn_dict = dict(gaussian=gaussian_kernel1d, cauchy=cauchy_kernel1d)

    def __init__(
        self,
        scales: Optional[List] = None,
        kernel: str = "gaussian",
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "MultiScaleLoss",
    ):
        """
        Init.

        :param scales: list of scalars or None, if None, do not apply any scaling.
        :param kernel: gaussian or cauchy.
        :param reduction: using SUM reduction over batch axis,
            this is for supporting multi-device training,
            and the loss will be divided by global batch size,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: str, name of the loss.
        """
        super().__init__(reduction=reduction, name=name)
        assert kernel in ["gaussian", "cauchy"]
        self.scales = scales
        self.kernel = kernel

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Use _call to calculate loss at different scales.

        :param y_true: ground-truth tensor, shape = (batch, dim1, dim2, dim3).
        :param y_pred: predicted tensor, shape = (batch, dim1, dim2, dim3).
        :return: multi-scale loss, shape = (batch, ).
        """
        if self.scales is None:
            return self._call(y_true=y_true, y_pred=y_pred)
        kernel_fn = self.kernel_fn_dict[self.kernel]
        losses = []
        for s in self.scales:
            if s == 0:
                # no smoothing
                losses.append(
                    self._call(
                        y_true=y_true,
                        y_pred=y_pred,
                    )
                )
            else:
                losses.append(
                    self._call(
                        y_true=separable_filter(
                            tf.expand_dims(y_true, axis=4), kernel_fn(s)
                        )[..., 0],
                        y_pred=separable_filter(
                            tf.expand_dims(y_pred, axis=4), kernel_fn(s)
                        )[..., 0],
                    )
                )
        loss = tf.add_n(losses)
        loss = loss / len(self.scales)
        return loss

    def _call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: ground-truth tensor.
        :param y_pred: predicted tensor.
        :return: negated loss.
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["scales"] = self.scales
        config["kernel"] = self.kernel
        return config


class DiceScore(MultiScaleLoss):
    """
    Define dice score.

    The formulation is:

        0. w_fg + w_bg = 1
        1. let y_prod = y_true * y_pred and y_sum  = y_true + y_pred
        2. num = 2 *  (w_fg * y_true * y_pred + w_bg * (1−y_true) * (1−y_pred))
               = 2 *  ((w_fg+w_bg) * y_prod - w_bg * y_sum + w_bg)
               = 2 *  (y_prod - w_bg * y_sum + w_bg)
        3. denom = (w_fg * (y_true + y_pred) + w_bg * (1−y_true + 1−y_pred))
                 = (w_fg-w_bg) * y_sum + 2 * w_bg
                 = (1-2*w_bg) * y_sum + 2 * w_bg
        4. dice score = num / denom

    where num and denom are summed over all axes except the batch axis.

    Reference:
        Sudre, Carole H., et al. "Generalised dice overlap as a deep learning loss
        function for highly unbalanced segmentations." Deep learning in medical image
        analysis and multimodal learning for clinical decision support.
        Springer, Cham, 2017. 240-248.
    """

    def __init__(
        self,
        binary: bool = False,
        background_weight: float = 0.0,
        smooth_nr: float = EPS,
        smooth_dr: float = EPS,
        scales: Optional[List] = None,
        kernel: str = "gaussian",
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "DiceScore",
    ):
        """
        Init.

        :param binary: if True, project y_true, y_pred to 0 or 1.
        :param background_weight: weight for background, where y == 0.
        :param smooth_nr: small constant added to numerator in case of zero covariance.
        :param smooth_dr: small constant added to denominator in case of zero variance.
        :param scales: list of scalars or None, if None, do not apply any scaling.
        :param kernel: gaussian or cauchy.
        :param reduction: using SUM reduction over batch axis,
            this is for supporting multi-device training,
            and the loss will be divided by global batch size,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: str, name of the loss.
        """
        super().__init__(scales=scales, kernel=kernel, reduction=reduction, name=name)
        if background_weight < 0 or background_weight > 1:
            raise ValueError(
                "The background weight for Dice Score must be "
                f"within [0, 1], got {background_weight}."
            )

        self.binary = binary
        self.background_weight = background_weight
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr
        self.flatten = tf.keras.layers.Flatten()

    def _call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        if self.binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        # (batch, ...) -> (batch, d)
        y_true = self.flatten(y_true)
        y_pred = self.flatten(y_pred)

        # for foreground class
        y_prod = tf.reduce_sum(y_true * y_pred, axis=1)
        y_sum = tf.reduce_sum(y_true + y_pred, axis=1)

        if self.background_weight > 0:
            # generalized
            vol = tf.reduce_sum(tf.ones_like(y_true), axis=1)
            numerator = 2 * (
                y_prod - self.background_weight * y_sum + self.background_weight * vol
            )
            denominator = (
                1 - 2 * self.background_weight
            ) * y_sum + 2 * self.background_weight * vol
        else:
            # foreground only
            numerator = 2 * y_prod
            denominator = y_sum

        return (numerator + self.smooth_nr) / (denominator + self.smooth_dr)

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(
            binary=self.binary,
            background_weight=self.background_weight,
            smooth_nr=self.smooth_nr,
            smooth_dr=self.smooth_dr,
        )
        return config


@REGISTRY.register_loss(name="dice")
class DiceLoss(NegativeLossMixin, DiceScore):
    """Revert the sign of DiceScore."""


@REGISTRY.register_loss(name="cross-entropy")
class CrossEntropy(MultiScaleLoss):
    """
    Define weighted cross-entropy.

    The formulation is:
        loss = − w_fg * y_true log(y_pred) - w_bg * (1−y_true) log(1−y_pred)
    """

    def __init__(
        self,
        binary: bool = False,
        background_weight: float = 0.0,
        smooth: float = EPS,
        scales: Optional[List] = None,
        kernel: str = "gaussian",
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "CrossEntropy",
    ):
        """
        Init.

        :param binary: if True, project y_true, y_pred to 0 or 1
        :param background_weight: weight for background, where y == 0.
        :param scales: list of scalars or None, if None, do not apply any scaling.
        :param kernel: gaussian or cauchy.
        :param reduction: using SUM reduction over batch axis,
            this is for supporting multi-device training,
            and the loss will be divided by global batch size,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: str, name of the loss.
        """
        super().__init__(scales=scales, kernel=kernel, reduction=reduction, name=name)
        if background_weight < 0 or background_weight > 1:
            raise ValueError(
                "The background weight for Cross Entropy must be "
                f"within [0, 1], got {background_weight}."
            )
        self.binary = binary
        self.background_weight = background_weight
        self.smooth = smooth
        self.flatten = tf.keras.layers.Flatten()

    def _call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        if self.binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        # (batch, ...) -> (batch, d)
        y_true = self.flatten(y_true)
        y_pred = self.flatten(y_pred)

        loss_fg = -tf.reduce_mean(y_true * tf.math.log(y_pred + self.smooth), axis=1)
        if self.background_weight > 0:
            loss_bg = -tf.reduce_mean(
                (1 - y_true) * tf.math.log(1 - y_pred + self.smooth), axis=1
            )
            return (
                1 - self.background_weight
            ) * loss_fg + self.background_weight * loss_bg
        else:
            return loss_fg

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(
            binary=self.binary,
            background_weight=self.background_weight,
            smooth=self.smooth,
        )
        return config


class JaccardIndex(DiceScore):
    """
    Define Jaccard index.

    The formulation is:
    1. num = y_true * y_pred
    2. denom = y_true + y_pred - y_true * y_pred
    3. Jaccard index = num / denom

        0. w_fg + w_bg = 1
        1. let y_prod = y_true * y_pred and y_sum  = y_true + y_pred
        2. num = (w_fg * y_true * y_pred + w_bg * (1−y_true) * (1−y_pred))
               = ((w_fg+w_bg) * y_prod - w_bg * y_sum + w_bg)
               = (y_prod - w_bg * y_sum + w_bg)
        3. denom = (w_fg * (y_true + y_pred - y_true * y_pred)
                  + w_bg * (1−y_true + 1−y_pred - (1−y_true) * (1−y_pred)))
                 = w_fg * (y_sum - y_prod) + w_bg * (1-y_prod)
                 = (1-w_bg) * y_sum - y_prod + w_bg
        4. dice score = num / denom
    """

    def __init__(
        self,
        binary: bool = False,
        background_weight: float = 0.0,
        smooth_nr: float = EPS,
        smooth_dr: float = EPS,
        scales: Optional[List] = None,
        kernel: str = "gaussian",
        reduction: str = tf.keras.losses.Reduction.NONE,
        name: str = "JaccardIndex",
    ):
        """
        Init.

        :param binary: if True, project y_true, y_pred to 0 or 1.
        :param background_weight: weight for background, where y == 0.
        :param smooth_nr: small constant added to numerator in case of zero covariance.
        :param smooth_dr: small constant added to denominator in case of zero variance.
        :param scales: list of scalars or None, if None, do not apply any scaling.
        :param kernel: gaussian or cauchy.
        :param reduction: using SUM reduction over batch axis,
            this is for supporting multi-device training,
            and the loss will be divided by global batch size,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: str, name of the loss.
        """
        super().__init__(
            binary=binary,
            background_weight=background_weight,
            smooth_nr=smooth_nr,
            smooth_dr=smooth_dr,
            scales=scales,
            kernel=kernel,
            reduction=reduction,
            name=name,
        )

    def _call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        if self.binary:
            y_true = tf.cast(y_true >= 0.5, dtype=y_true.dtype)
            y_pred = tf.cast(y_pred >= 0.5, dtype=y_pred.dtype)

        # (batch, ...) -> (batch, d)
        y_true = self.flatten(y_true)
        y_pred = self.flatten(y_pred)

        # for foreground class
        y_prod = tf.reduce_sum(y_true * y_pred, axis=1)
        y_sum = tf.reduce_sum(y_true + y_pred, axis=1)

        if self.background_weight > 0:
            # generalized
            vol = tf.reduce_sum(tf.ones_like(y_true), axis=1)
            numerator = (
                y_prod - self.background_weight * y_sum + self.background_weight * vol
            )
            denominator = (
                (1 - self.background_weight) * y_sum
                - y_prod
                + self.background_weight * vol
            )
        else:
            # foreground only
            numerator = y_prod
            denominator = y_sum - y_prod

        return (numerator + self.smooth_nr) / (denominator + self.smooth_dr)


@REGISTRY.register_loss(name="jaccard")
class JaccardLoss(NegativeLossMixin, JaccardIndex):
    """Revert the sign of JaccardIndex."""


def compute_centroid(mask: tf.Tensor, grid: tf.Tensor) -> tf.Tensor:
    """
    Calculate the centroid of the mask.
    :param mask: shape = (batch, dim1, dim2, dim3)
    :param grid: shape = (1, dim1, dim2, dim3, 3)
    :return: shape = (batch, 3), batch of vectors denoting
             location of centroids.
    """
    assert len(mask.shape) == 4
    assert len(grid.shape) == 5
    bool_mask = tf.expand_dims(
        tf.cast(mask >= 0.5, dtype=tf.float32), axis=4
    )  # (batch, dim1, dim2, dim3, 1)
    masked_grid = bool_mask * grid  # (batch, dim1, dim2, dim3, 3)
    numerator = tf.reduce_sum(masked_grid, axis=[1, 2, 3])  # (batch, 3)
    denominator = tf.reduce_sum(bool_mask, axis=[1, 2, 3])  # (batch, 1)
    return (numerator + EPS) / (denominator + EPS)  # (batch, 3)


def compute_centroid_distance(
    y_true: tf.Tensor, y_pred: tf.Tensor, grid: tf.Tensor
) -> tf.Tensor:
    """
    Calculate the L2-distance between two tensors' centroids.
    :param y_true: tensor, shape = (batch, dim1, dim2, dim3)
    :param y_pred: tensor, shape = (batch, dim1, dim2, dim3)
    :param grid: tensor, shape = (1, dim1, dim2, dim3, 3)
    :return: shape = (batch,)
    """
    centroid_1 = compute_centroid(mask=y_pred, grid=grid)  # (batch, 3)
    centroid_2 = compute_centroid(mask=y_true, grid=grid)  # (batch, 3)
    return tf.sqrt(tf.reduce_sum((centroid_1 - centroid_2) ** 2, axis=1))


def foreground_proportion(y: tf.Tensor) -> tf.Tensor:
    """
    Calculate the percentage of foreground vs background per 3d volume.
    :param y: shape = (batch, dim1, dim2, dim3), a 3D label tensor
    :return: shape = (batch,)
    """
    y = tf.cast(y >= 0.5, dtype=tf.float32)
    return tf.reduce_sum(y, axis=[1, 2, 3]) / tf.reduce_sum(
        tf.ones_like(y), axis=[1, 2, 3]
    )
