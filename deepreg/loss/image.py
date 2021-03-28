"""Provide different loss or metrics classes for images."""
import numpy as np
import tensorflow as tf

from deepreg.constant import EPS
from deepreg.loss.util import NegativeLossMixin
from deepreg.loss.util import gaussian_kernel1d_size as gaussian_kernel1d
from deepreg.loss.util import (
    rectangular_kernel1d,
    separable_filter,
    triangular_kernel1d,
)
from deepreg.registry import REGISTRY


@REGISTRY.register_loss(name="ssd")
class SumSquaredDifference(tf.keras.losses.Loss):
    """
    Sum of squared distance between y_true and y_pred.

    y_true and y_pred have to be at least 1d tensor, including batch axis.
    """

    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "SumSquaredDifference",
    ):
        """
        Init.

        :param reduction: using SUM reduction over batch axis,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        """
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """
        loss = tf.math.squared_difference(y_true, y_pred)
        loss = tf.keras.layers.Flatten()(loss)
        return tf.reduce_mean(loss, axis=1)


class GlobalMutualInformation(tf.keras.losses.Loss):
    """
    Differentiable global mutual information via Parzen windowing method.

    y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference: https://dspace.mit.edu/handle/1721.1/123142,
        Section 3.1, equation 3.1-3.5, Algorithm 1
    """

    def __init__(
        self,
        num_bins: int = 23,
        sigma_ratio: float = 0.5,
        vmin: float = 0.0,
        vmax: float = 1.0,
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "GlobalMutualInformation",
    ):
        """
        Init.

        :param num_bins: number of bins for intensity, the default value is empirical.
        :param sigma_ratio: sigma in used Gaussian equals bin_size * sigma_ratio.
        :param reduction: using SUM reduction over batch axis,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        """
        super().__init__(reduction=reduction, name=name)
        if vmax <= vmin:
            raise ValueError(
                f"vmax must be greater than vmin, " f"got vmax = {vmax}, vmin = {vmin}."
            )
        # arguments
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio
        self.vmin = vmin
        self.vmax = vmax

        # constants to be used
        # sigma in the Gaussian function (weighting function W)
        sigma = (vmax - vmin) * sigma_ratio
        if sigma <= EPS:
            raise ValueError(f"The sigma in Gaussian is too small, " f"got {sigma}")
        self.denom = 2 * sigma * sigma

        # intensity is split into bins between vmin and vmax
        self.bin_centers = tf.linspace(
            self.vmin, self.vmax, self.num_bins
        )  # (num_bins,)
        self.bin_centers = self.bin_centers[None, None, ...]  # (1, 1, num_bins)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :param y_pred: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :return: shape = (batch,)
        """
        # adjust tensor shape
        if len(y_true.shape) == 4:
            y_true = tf.expand_dims(y_true, axis=4)
            y_pred = tf.expand_dims(y_pred, axis=4)
        assert len(y_true.shape) == len(y_pred.shape) == 5

        batch_size = y_true.shape[0]
        num_voxels = tf.cast(np.prod(y_true.shape[1:4]), dtype=y_true.dtype)

        # (batch, num_voxels, 1)
        y_true = tf.reshape(y_true, (batch_size, -1, 1))
        y_pred = tf.reshape(y_true, (batch_size, -1, 1))

        wa = self.weight(y_true)  # (batch, num_voxels, num_bins)
        wb = self.weight(y_pred)  # (batch, num_voxels, num_bins)
        wa = tf.transpose(wa, perm=[0, 2, 1])  # (batch, num_bins, num_voxels)

        # pa pb should be a proba distribution
        pa = tf.reduce_mean(wa, axis=2, keepdims=True)  # (batch, num_bins, 1)
        pb = tf.reduce_mean(wb, axis=1, keepdims=True)  # (batch, 1, num_bins)

        # both papb and pab have shape = (batch, num_bins, num_bins)
        # pab should be a proba distribution
        papb = tf.matmul(pa, pb)
        pab = tf.matmul(wa, wb)
        pab /= num_voxels

        div = (pab + EPS) / (papb + EPS)
        mi = tf.reduce_sum(pab * tf.math.log(div + EPS), axis=[1, 2])

        return mi

    def weight(self, x: tf.Tensor) -> tf.Tensor:
        """Calculate weight using Parzen windowing.

        The weight is only exp(...) without being divided by sigma * (2 * pi) ** 0.5.
        As the weight will be later renormalized, division here is not necessary.

        :param x: image tensor, shape = (batch, nb_voxels, 1).
        :return: W(x) as equation 3.2 in the reference, but with a constant difference
            shape = (batch, nb_voxels, num_bins)
        """
        # each voxel contributes continuously to a range of histogram bin
        # (batch, nb_voxels, num_bins)
        bin_centers = tf.cast(self.bin_centers, x.dtype)
        denom = tf.cast(self.denom, x.dtype)
        logits = -((x - bin_centers) ** 2) / denom
        w = tf.math.softmax(logits=logits, axis=-1)

        return w

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["num_bins"] = self.num_bins
        config["sigma_ratio"] = self.sigma_ratio
        config["vmin"] = self.vmin
        config["vmax"] = self.vmax
        return config


@REGISTRY.register_loss(name="gmi")
class GlobalMutualInformationLoss(NegativeLossMixin, GlobalMutualInformation):
    """Revert the sign of GlobalMutualInformation."""


class LocalNormalizedCrossCorrelation(tf.keras.losses.Loss):
    """
    Local squared zero-normalized cross-correlation.

    Denote y_true as t and y_pred as p. Consider a window having n elements.
    Each position in the window corresponds a weight w_i for i=1:n.

    Define the discrete expectation in the window E[t] as

        E[t] = sum_i(w_i * t_i) / sum_i(w_i)

    Here, we assume sum_i(w_i) == 1, means the weights have been normalized.

    Similarly, the discrete variance in the window V[t] is

        V[t] = E[(t - E[t])**2]

    The local squared zero-normalized cross-correlation is therefore

        E[ (t-E[t]) * (p-E[p]) ] ** 2 / V[t] / V[p]

    When calculating variance, we choose to subtract the mean first then calculte
    variance instead of calculating E[t**2] - E[t] ** 2, the reason is that when
    E[t**2] and E[t] ** 2 are both very large or very small, the subtraction may
    have large rounding error and makes the result inaccurate. Also, it is not
    guaranteed that the result >= 0. For more details, please read "Algorithms for
    computing the sample variance: Analysis and recommendations." page 1.

    For now, y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference:

        - Zero-normalized cross-correlation (ZNCC):
            https://en.wikipedia.org/wiki/Cross-correlation
        - https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
        - Chan, Tony F., Gene H. Golub, and Randall J. LeVeque.
          "Algorithms for computing the sample variance: Analysis and recommendations."
           The American Statistician 37.3 (1983): 242-247.
    """

    kernel_fn_dict = dict(
        gaussian=gaussian_kernel1d,
        rectangular=rectangular_kernel1d,
        triangular=triangular_kernel1d,
    )

    def __init__(
        self,
        kernel_size: int = 9,
        kernel_type: str = "rectangular",
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "LocalNormalizedCrossCorrelation",
    ):
        """
        Init.

        :param kernel_size: int. Kernel size or kernel sigma for kernel_type='gauss'.
        :param kernel_type: str, rectangular, triangular or gaussian
        :param reduction: using SUM reduction over batch axis,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        """
        super().__init__(reduction=reduction, name=name)
        if kernel_type not in self.kernel_fn_dict.keys():
            raise ValueError(
                f"Wrong kernel_type {kernel_type} for LNCC loss type. "
                f"Feasible values are {self.kernel_fn_dict.keys()}"
            )
        self.kernel_fn = self.kernel_fn_dict[kernel_type]
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size

        # (kernel_size, )
        # sum of the kernel weights would be one
        self.kernel = self.kernel_fn(kernel_size=self.kernel_size)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :param y_pred: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :return: shape = (batch,)
        """
        # adjust shape to be (batch, dim1, dim2, dim3, ch)
        if len(y_true.shape) == 4:
            y_true = tf.expand_dims(y_true, axis=4)
            y_pred = tf.expand_dims(y_pred, axis=4)
        assert len(y_true.shape) == len(y_pred.shape) == 5

        # t = y_true, p = y_pred
        t_mean = separable_filter(y_true, kernel=self.kernel)
        p_mean = separable_filter(y_pred, kernel=self.kernel)

        t = y_true - t_mean
        p = y_pred - p_mean

        # the variance can be biased but as both num and denom are biased
        # it got cancelled
        # https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Reliability_weights
        cross = separable_filter(t * p, kernel=self.kernel)
        t_var = separable_filter(t * t, kernel=self.kernel)
        p_var = separable_filter(p * p, kernel=self.kernel)

        num = cross * cross
        denom = t_var * p_var
        ncc = (num + EPS) / (denom + EPS)

        return tf.reduce_mean(ncc, axis=[1, 2, 3, 4])

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["kernel_size"] = self.kernel_size
        config["kernel_type"] = self.kernel_type
        return config


@REGISTRY.register_loss(name="lncc")
class LocalNormalizedCrossCorrelationLoss(
    NegativeLossMixin, LocalNormalizedCrossCorrelation
):
    """Revert the sign of LocalNormalizedCrossCorrelation."""


class GlobalNormalizedCrossCorrelation(tf.keras.losses.Loss):
    """
    Global squared zero-normalized cross-correlation.

    Compute the squared cross-correlation between the reference and moving images
    y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference:

        - Zero-normalized cross-correlation (ZNCC):
            https://en.wikipedia.org/wiki/Cross-correlation

    """

    def __init__(
        self,
        reduction: str = tf.keras.losses.Reduction.AUTO,
        name: str = "GlobalNormalizedCrossCorrelation",
    ):
        """
        Init.
        :param reduction: using AUTO reduction,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        """
        super().__init__(reduction=reduction, name=name)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, ...)
        :param y_pred: shape = (batch, ...)
        :return: shape = (batch,)
        """

        axis = [a for a in range(1, len(y_true.shape))]
        mu_pred = tf.reduce_mean(y_pred, axis=axis, keepdims=True)
        mu_true = tf.reduce_mean(y_true, axis=axis, keepdims=True)
        var_pred = tf.math.reduce_variance(y_pred, axis=axis)
        var_true = tf.math.reduce_variance(y_true, axis=axis)
        numerator = tf.abs(
            tf.reduce_mean((y_pred - mu_pred) * (y_true - mu_true), axis=axis)
        )

        return (numerator * numerator + EPS) / (var_pred * var_true + EPS)


@REGISTRY.register_loss(name="gncc")
class GlobalNormalizedCrossCorrelationLoss(
    NegativeLossMixin, GlobalNormalizedCrossCorrelation
):
    """Revert the sign of GlobalNormalizedCrossCorrelation."""
