# flake8: noqa
from deepreg.model.loss.deform import BendingEnergy, GradientNorm
from deepreg.model.loss.image import (
    GlobalMutualInformation,
    GlobalMutualInformationLoss,
    LocalNormalizedCrossCorrelation,
    LocalNormalizedCrossCorrelationLoss,
    SumSquaredDifference,
)
from deepreg.model.loss.label import (
    CrossEntropy,
    DiceLoss,
    DiceScore,
    JaccardIndex,
    JaccardLoss,
)
