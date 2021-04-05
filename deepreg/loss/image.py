"""Provide different loss or metrics classes for images."""
import tensorflow as tf

from deepreg.constant import EPS
from deepreg.loss.kernel import gaussian_kernel1d_size as gaussian_kernel1d
from deepreg.loss.kernel import rectangular_kernel1d, triangular_kernel1d
from deepreg.loss.util import NegativeLossMixin, separable_filter
from deepreg.registry import REGISTRY


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
        name: str = "GlobalMutualInformation",
        **kwargs,
    ):
        """
        Init.

        :param num_bins: number of bins for intensity, the default value is empirical.
        :param sigma_ratio: a hyper param for gaussian function
        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
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

    Similarly, the discrete variance in the window V[t] is

        V[t] = E[t**2] - E[t] ** 2

    The local squared zero-normalized cross-correlation is therefore

        E[ (t-E[t]) * (p-E[p]) ] ** 2 / V[t] / V[p]

    where the expectation in numerator is

        E[ (t-E[t]) * (p-E[p]) ] = E[t * p] - E[t] * E[p]

    Different kernel corresponds to different weights.

    For now, y_true and y_pred have to be at least 4d tensor, including batch axis.

    Reference:

        - Zero-normalized cross-correlation (ZNCC):
            https://en.wikipedia.org/wiki/Cross-correlation
        - Code: https://github.com/voxelmorph/voxelmorph/blob/legacy/src/losses.py
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
        smooth_nr: float = EPS,
        smooth_dr: float = EPS,
        name: str = "LocalNormalizedCrossCorrelation",
        **kwargs,
    ):
        """
        Init.

        :param kernel_size: int. Kernel size or kernel sigma for kernel_type='gauss'.
        :param kernel_type: str, rectangular, triangular or gaussian
        :param smooth_nr: small constant added to numerator in case of zero covariance.
        :param smooth_dr: small constant added to denominator in case of zero variance.
        :param name: name of the loss.
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)
        if kernel_type not in self.kernel_fn_dict.keys():
            raise ValueError(
                f"Wrong kernel_type {kernel_type} for LNCC loss type. "
                f"Feasible values are {self.kernel_fn_dict.keys()}"
            )
        self.kernel_fn = self.kernel_fn_dict[kernel_type]
        self.kernel_type = kernel_type
        self.kernel_size = kernel_size
        self.smooth_nr = smooth_nr
        self.smooth_dr = smooth_dr

        # (kernel_size, )
        self.kernel = self.kernel_fn(kernel_size=self.kernel_size)
        # E[1] = sum_i(w_i), ()
        self.kernel_vol = tf.reduce_sum(
            self.kernel[:, None, None]
            * self.kernel[None, :, None]
            * self.kernel[None, None, :]
        )

    def calc_ncc(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return NCC for a batch.

        The kernel should not be normalized, as normalizing them leads to computation
        with small values and the precision will be reduced.
        Here both numerator and denominator are actually multiplied by kernel volume,
        which helps the precision as well.
        However, when the variance is zero, the obtained value might be negative due to
        machine error. Therefore a hard-coded clipping is added to
        prevent division by zero.

        :param y_true: shape = (batch, dim1, dim2, dim3, 1)
        :param y_pred: shape = (batch, dim1, dim2, dim3, 1)
        :return: shape = (batch, dim1, dim2, dim3. 1)
        """

        # t = y_true, p = y_pred
        # (batch, dim1, dim2, dim3, 1)
        t2 = y_true * y_true
        p2 = y_pred * y_pred
        tp = y_true * y_pred

        # sum over kernel
        # (batch, dim1, dim2, dim3, 1)
        t_sum = separable_filter(y_true, kernel=self.kernel)  # E[t] * E[1]
        p_sum = separable_filter(y_pred, kernel=self.kernel)  # E[p] * E[1]
        t2_sum = separable_filter(t2, kernel=self.kernel)  # E[tt] * E[1]
        p2_sum = separable_filter(p2, kernel=self.kernel)  # E[pp] * E[1]
        tp_sum = separable_filter(tp, kernel=self.kernel)  # E[tp] * E[1]

        # average over kernel
        # (batch, dim1, dim2, dim3, 1)
        t_avg = t_sum / self.kernel_vol  # E[t]
        p_avg = p_sum / self.kernel_vol  # E[p]

        # shape = (batch, dim1, dim2, dim3, 1)
        cross = tp_sum - p_avg * t_sum  # E[tp] * E[1] - E[p] * E[t] * E[1]
        t_var = t2_sum - t_avg * t_sum  # V[t] * E[1]
        p_var = p2_sum - p_avg * p_sum  # V[p] * E[1]

        # ensure variance >= 0
        t_var = tf.maximum(t_var, 0)
        p_var = tf.maximum(p_var, 0)

        # (E[tp] - E[p] * E[t]) ** 2 / V[t] / V[p]
        ncc = (cross * cross + self.smooth_nr) / (t_var * p_var + self.smooth_dr)

        return ncc

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Return loss for a batch.

        TODO: support channel axis dimension > 1.

        :param y_true: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, 1)
        :param y_pred: shape = (batch, dim1, dim2, dim3)
            or (batch, dim1, dim2, dim3, 1)
        :return: shape = (batch,)
        """
        # sanity checks
        if len(y_true.shape) == 4:
            y_true = tf.expand_dims(y_true, axis=4)
        if y_true.shape[4] != 1:
            raise ValueError(
                "Last dimension of y_true is not one. " f"y_true.shape = {y_true.shape}"
            )
        if len(y_pred.shape) == 4:
            y_pred = tf.expand_dims(y_pred, axis=4)
        if y_pred.shape[4] != 1:
            raise ValueError(
                "Last dimension of y_pred is not one. " f"y_pred.shape = {y_pred.shape}"
            )

        ncc = self.calc_ncc(y_true=y_true, y_pred=y_pred)
        return tf.reduce_mean(ncc, axis=[1, 2, 3, 4])

    def get_config(self) -> dict:
        """Return the config dictionary for recreating this class."""
        config = super().get_config()
        config.update(
            kernel_size=self.kernel_size,
            kernel_type=self.kernel_type,
            smooth_nr=self.smooth_nr,
            smooth_dr=self.smooth_dr,
        )
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
        name: str = "GlobalNormalizedCrossCorrelation",
        **kwargs,
    ):
        """
        Init.

        :param name: name of the loss
        :param kwargs: additional arguments.
        """
        super().__init__(name=name, **kwargs)

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
