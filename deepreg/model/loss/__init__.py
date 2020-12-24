# flake8: noqa
from deepreg.model.loss.image import (
    GlobalMutualInformation3D,
    GlobalMutualInformation3DLoss,
    LocalNormalizedCrossCorrelation3D,
    LocalNormalizedCrossCorrelation3DLoss,
    SumSquaredDistance,
)
from deepreg.model.loss.label import (
    CrossEntropy,
    DiceLoss,
    DiceScore,
    JaccardIndex,
    JaccardLoss,
)
