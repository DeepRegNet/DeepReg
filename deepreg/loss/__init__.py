"""Define different loss classes for image, label and regularization."""
# flake8: noqa
from deepreg.loss.deform import BendingEnergy, GradientNorm
from deepreg.loss.image import (
    GlobalMutualInformation,
    GlobalMutualInformationLoss,
    LocalNormalizedCrossCorrelation,
    LocalNormalizedCrossCorrelationLoss,
)
from deepreg.loss.label import (
    CrossEntropy,
    DiceLoss,
    DiceScore,
    JaccardIndex,
    JaccardLoss,
    SumSquaredDifference,
)
