"""Provide different loss or metrics classes for images."""
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
        reduction: str = tf.keras.losses.Reduction.SUM,
        name: str = "GlobalMutualInformation",
    ):
        """
        Init.

        :param num_bins: number of bins for intensity, the default value is empirical.
        :param sigma_ratio: a hyper param for gaussian function
        :param reduction: using SUM reduction over batch axis,
            calling the loss like `loss(y_true, y_pred)` will return a scalar tensor.
        :param name: name of the loss
        """
        super().__init__(reduction=reduction, name=name)
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        :param y_true: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :param y_pred: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, ch)
        :return: shape = (batch,)
        """
        # adjust
        if len(y_true.shape) == 4:
            y_true = tf.expand_dims(y_true, axis=4)
            y_pred = tf.expand_dims(y_pred, axis=4)
        assert len(y_true.shape) == len(y_pred.shape) == 5

        # intensity is split into bins between 0, 1
        y_true = tf.clip_by_value(y_true, 0, 1)
        y_pred = tf.clip_by_value(y_pred, 0, 1)
        bin_centers = tf.linspace(0.0, 1.0, self.num_bins)  # (num_bins,)
        bin_centers = tf.cast(bin_centers, dtype=y_true.dtype)
        bin_centers = bin_centers[None, None, ...]  # (1, 1, num_bins)
        sigma = (
            tf.reduce_mean(bin_centers[:, :, 1:] - bin_centers[:, :, :-1])
            * self.sigma_ratio
        )  # scalar, sigma in the Gaussian function (weighting function W)
        preterm = 1 / (2 * tf.math.square(sigma))  # scalar
        batch, w, h, z, c = y_true.shape
        y_true = tf.reshape(y_true, [batch, w * h * z * c, 1])  # (batch, nb_voxels, 1)
        y_pred = tf.reshape(y_pred, [batch, w * h * z * c, 1])  # (batch, nb_voxels, 1)
        nb_voxels = y_true.shape[1] * 1.0  # w * h * z, number of voxels

        # each voxel contributes continuously to a range of histogram bin
        ia = tf.math.exp(
            -preterm * tf.math.square(y_true - bin_centers)
        )  # (batch, nb_voxels, num_bins)
        ia /= tf.reduce_sum(ia, -1, keepdims=True)  # (batch, nb_voxels, num_bins)
        ia = tf.transpose(ia, (0, 2, 1))  # (batch, num_bins, nb_voxels)
        pa = tf.reduce_mean(ia, axis=-1, keepdims=True)  # (batch, num_bins, 1)

        ib = tf.math.exp(
            -preterm * tf.math.square(y_pred - bin_centers)
        )  # (batch, nb_voxels, num_bins)
        ib /= tf.reduce_sum(ib, -1, keepdims=True)  # (batch, nb_voxels, num_bins)
        pb = tf.reduce_mean(ib, axis=1, keepdims=True)  # (batch, 1, num_bins)

        papb = tf.matmul(pa, pb)  # (batch, num_bins, num_bins)
        pab = tf.matmul(ia, ib)  # (batch, num_bins, num_bins)
        pab /= nb_voxels

        # MI: sum(P_ab * log(P_ab/P_ap_b))
        div = (pab + EPS) / (papb + EPS)
        return tf.reduce_sum(pab * tf.math.log(div + EPS), axis=[1, 2])

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config["num_bins"] = self.num_bins
        config["sigma_ratio"] = self.sigma_ratio
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
