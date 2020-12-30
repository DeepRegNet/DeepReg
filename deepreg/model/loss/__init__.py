# flake8: noqa
from deepreg.model.loss.deform import BendingEnergy3D, GradientNorm3D
from deepreg.model.loss.image import (
    GlobalMutualInformation3D,
    GlobalMutualInformation3DLoss,
    LocalNormalizedCrossCorrelation3D,
    LocalNormalizedCrossCorrelation3DLoss,
    SumSquaredDifference,
)
from deepreg.model.loss.label import (
    CrossEntropy,
    DiceLoss,
    DiceScore,
    JaccardIndex,
    JaccardLoss,
)
