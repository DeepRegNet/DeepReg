# Registered Classes

> This file is generated automatically.

The following tables contain all registered classes with their categories and keys.

## Backbone

The category is `backbone_class`. Registered keys and values are as following.

| key      | value                                         |
| :------- | :-------------------------------------------- |
| "global" | `deepreg.model.backbone.global_net.GlobalNet` |
| "local"  | `deepreg.model.backbone.local_net.LocalNet`   |
| "unet"   | `deepreg.model.backbone.u_net.UNet`           |

## Model

The category is `model_class`. Registered keys and values are as following.

| key           | value                                    |
| :------------ | :--------------------------------------- |
| "conditional" | `deepreg.model.network.ConditionalModel` |
| "ddf"         | `deepreg.model.network.DDFModel`         |
| "dvf"         | `deepreg.model.network.DVFModel`         |

## Loss

The category is `loss_class`. Registered keys and values are as following.

| key             | value                                                    |
| :-------------- | :------------------------------------------------------- |
| "bending"       | `deepreg.loss.deform.BendingEnergy`                      |
| "cross-entropy" | `deepreg.loss.label.CrossEntropy`                        |
| "dice"          | `deepreg.loss.label.DiceLoss`                            |
| "gmi"           | `deepreg.loss.image.GlobalMutualInformationLoss`         |
| "gradient"      | `deepreg.loss.deform.GradientNorm`                       |
| "jaccard"       | `deepreg.loss.label.JaccardLoss`                         |
| "lncc"          | `deepreg.loss.image.LocalNormalizedCrossCorrelationLoss` |
| "ssd"           | `deepreg.loss.image.SumSquaredDifference`                |

## Data Augmentation

The category is `da_class`. Registered keys and values are as following.

| key      | value                                                |
| :------- | :--------------------------------------------------- |
| "affine" | `deepreg.dataset.preprocess.RandomAffineTransform3D` |
| "ddf"    | `deepreg.dataset.preprocess.RandomDDFTransform3D`    |
