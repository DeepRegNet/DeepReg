# Custom loss function

This tutorial will take an image intensity based loss function (mutual information) as
an example to show how to add a new loss to DeepReg.

## A brief review of the types of loss functions in DeepReg

Three main types of the loss functions are supported in DeepReg:
`intensity (image) based loss`, `label based loss` and `deformation loss`. See
[Docs here](registration.html#loss) for details. The corresponding source files for the
losses is included in `deepreg/model/loss`.

## Step 1: Add the new function in loss source code

The first step is to add your own loss function, which should take at least 2
parameters, `y_true` for the ground truth and `y_pred` for the prediction. e.g. in
`deepreg/model/loss/image.py`. The loss can be defined as:

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

In order to be compatible with the pipeline in DeepReg, another modification is needed
in `deeepreg/model/loss/image.py`. The modification is simply adding an `elif` branch to
the `dissimilarity_fn` function. The following code block shows the added `elif` branch
where we use `"gmi"` to represent the global mutual information:

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

## Step 2: Add test functions (for contributing developers, optional for users)

Add corresponding unit test for the new added functions to `deepreg/test/unit`. This is
optional for the users. Everyone is warmly welcome to make contribution to DeepReg.
Please follow our [contribution guidelines](../contributing/code.html) here.

## Step 3: Set yaml configuration files

We take the
[paired prostate MR and Ultrasound registration demo](../../../demos/paired_mrus_prostate)
as an example. In order to use the newly added loss, all that is needed is to modify the
loss configuration in the train configuration file `paired_mrus_prostate_train.yaml`
(lines 10-15):

```
# define the loss function for training
  loss:
    dissimilarity:
      image:
        name: "gmi"
        weight: 1.0
```

That's it. Follow the instructions in the demo to begin training with the newly added
loss.
