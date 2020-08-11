# Customize your own loss functions

This tutorial will take an image intensity based loss function (Mutual information) as an example to show how to add a new loss to deepreg.

## Intruductions to the types of loss functions in deepreg

Three main types of the loss functions are supported in deepreg: `intensity (image) based loss`, `label based loss` and `deformation loss`. See [Docs here](https://deepregnet.github.io/DeepReg/#/tutorial_registration?id=loss) for details. The correspondng source files for the losses is included in `deepreg/model/loss`.

## Step 1: Add new functions in loss sources codes

The first step is to add your own loss functions, which should takes at least 2 parameters, `y_true` for the groud truth and `y_pred` for the prediction. 
e.g. in `deepreg/model/loss/image.py`
```
def global_mutual_information(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    differentiable global mutual information loss via Parzen windowing method.
    reference: https://dspace.mit.edu/handle/1721.1/123142, Section 3.1, equation 3.1-3.5
    :y_true: shape = (batch, dim1, dim2, dim3, ch)
    :y_pred: shape = (batch, dim1, dim2, dim3, ch)
    :return: shape = (batch,)
    """
    ...
    return tf.reduce_sum(pab * tf.math.log(pab / papb + eps), axis=[1, 2])
```

In order to be compatible with the pipeline in deepreg, another modification is need as well in `deeepreg/model/loss/image.py`, which is just need to add an `elif` branch. We use `"gmi"` here to represent the global mutual information:
```
def dissimilarity_fn(
    y_true: tf.Tensor, y_pred: tf.Tensor, name: str, **kwargs
) -> tf.Tensor:
    """
    :param y_true: fixed_image, shape = (batch, f_dim1, f_dim2, f_dim3)
    :param y_pred: warped_moving_image, shape = (batch, f_dim1, f_dim2, f_dim3)
    :param name: name of the dissimilarity function
    :param kwargs: absorb additional parameters
    :return: shape = (batch,)
    """
    assert name in ["lncc", "ssd", "gmi"]
    # shape = (batch, f_dim1, f_dim2, f_dim3, 1)
    y_true = tf.expand_dims(y_true, axis=4)
    y_pred = tf.expand_dims(y_pred, axis=4)
    if name == "lncc":
        return -local_normalized_cross_correlation(y_true, y_pred, **kwargs)
    elif name == "ssd":
        return ssd(y_true, y_pred)
    elif name == "gmi":
        return -global_mutual_information(y_true, y_pred)
    else:
raise ValueError("Unknown loss type.")
```
## Step 2: Add test functions (For developers, optional for users)

Add correspoding unit test for the new added functions to `deepreg/test/unit`. It's just optional for the users. We welcome everyone who wants to make contribution to deepreg. Please follow our [contribution guidelines](https://github.com/DeepRegNet/DeepReg/blob/20-mutual-information/docs/CONTRIBUTING.md) here.

## Step 3: Set yaml configuration files

We take the [paired prostate MR and Ultrasound registration demo](https://github.com/DeepRegNet/DeepReg/tree/20-mutual-information/demos/paired_mrus_prostate) as an example, in order to use the new added loss, all needed to do is to modify the loss configures in the train yaml file `paired_mrus_prostate_train.yaml`, line 10-15:
```
# define the loss function for training
  loss:
    dissimilarity:
      image:
        name: "gmi"
        weight: 1.0
```
And then just follow the instructions in the demo to begin the training with the new added loss.
