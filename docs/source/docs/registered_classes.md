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

| key             | value                                                     |
| :-------------- | :-------------------------------------------------------- |
| "bending"       | `deepreg.loss.deform.BendingEnergy`                       |
| "cross-entropy" | `deepreg.loss.label.CrossEntropyLoss`                     |
| "dice"          | `deepreg.loss.label.DiceLoss`                             |
| "gmi"           | `deepreg.loss.image.GlobalMutualInformationLoss`          |
| "gncc"          | `deepreg.loss.image.GlobalNormalizedCrossCorrelationLoss` |
| "gradient"      | `deepreg.loss.deform.GradientNorm`                        |
| "jaccard"       | `deepreg.loss.label.JaccardLoss`                          |
| "lncc"          | `deepreg.loss.image.LocalNormalizedCrossCorrelationLoss`  |
| "ssd"           | `deepreg.loss.label.SumSquaredDifferenceLoss`             |

## Data Augmentation

The category is `da_class`. Registered keys and values are as following.

| key      | value                                                |
| :------- | :--------------------------------------------------- |
| "affine" | `deepreg.dataset.preprocess.RandomAffineTransform3D` |
| "ddf"    | `deepreg.dataset.preprocess.RandomDDFTransform3D`    |

## Data Loader

The category is `data_loader_class`. Registered keys and values are as following.

| key        | value                                                       |
| :--------- | :---------------------------------------------------------- |
| "grouped"  | `deepreg.dataset.loader.grouped_loader.GroupedDataLoader`   |
| "paired"   | `deepreg.dataset.loader.paired_loader.PairedDataLoader`     |
| "unpaired" | `deepreg.dataset.loader.unpaired_loader.UnpairedDataLoader` |

## File Loader

The category is `file_loader_class`. Registered keys and values are as following.

| key     | value                                                 |
| :------ | :---------------------------------------------------- |
| "h5"    | `deepreg.dataset.loader.h5_loader.H5FileLoader`       |
| "nifti" | `deepreg.dataset.loader.nifti_loader.NiftiFileLoader` |
