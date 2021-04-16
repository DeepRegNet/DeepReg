# Configuration File

In addition to the arguments provided to the command line tools, detailed training and
prediction configuration is specified in a `YAML` file. The configuration file contains
two sections, `dataset` and `train`. Within `dataset` one specifies the data file
formats, sizes, as well as the data loader to use. The `train` section specifies
parameters related to the neural network.

## Dataset section

The `dataset` section specifies the path to the data to be used during training, the
data loader to use as well as the specific arguments to configure the data loader.

### Split keys - Required

The data paths, data format, and label availability of the training, validation and
testing data are specified under the corresponding sections separately:

```yaml
dataset:
  train:
    dir: "data/test/h5/paired/train"
    format: "h5"
    labeled: true
  valid:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
```

For data paths, multiple dataset directories can be specified, such that data are
sampled across several folders:

```yaml
dataset:
  train:
    dir:
      - "data/test/h5/paired/train1"
      - "data/test/h5/paired/train2"
    format: "h5"
    labeled: true
  valid:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
```

For data format, different formats requires different file structure and format. Thus
different file loader will be used. Check the
[data loader configuration](dataset_loader.html) for more details.

Currently, DeepReg file loaders support Nifti and H5 file types - alternate file formats
will raise errors in the data loaders. To indicate which format to use, pass a string to
this field as either "nifti" or "h5":

```yaml
dataset:
  train:
    dir:
      - "data/test/nifti/paired/train1"
      - "data/test/nifti/paired/train2"
    format: "nifti"
    labeled: true
```

The `labeled` key indicates whether segmentation labels are available for training or
evaluation. Use `true` and `false` to indicate the availability and unavailability
correspondingly. In particular, if the value passed is false, the labels will not be
used even if they are available in the associated directories.

```yaml
dataset:
  train:
    dir:
      - "data/test/nifti/paired/train1"
      - "data/test/nifti/paired/train2"
    format: "nifti"
    labeled: false # labels are not available
```

### Type key - Required

The type of data loader used will depend on how one wants to train the network.
Currently, DeepReg data loaders support the `paired`, `unpaired`, and `grouped` training
strategies. Passing a string that doesn't match any of the above would raise an error.
The data loader type would be specified using the `type` key:

```yaml
dataset:
  train:
    dir: "data/test/h5/paired/train"
    format: "h5"
    labeled: true
  valid:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  type: "paired" # one of "paired", "unpaired" or "grouped"
```

### Data loader dependent keys

Depending on which string is passed to the `type` key, DeepReg will initialize a
different data loader instance with different sampling strategies. These are described
in depth in the [dataset loader configuration](dataset_loader.html) documentation. Here
we outline the arguments necessary to configure the different data loaders.

#### Sample_label - Required

In the case that we have more than one label per image, we need to inform the loader
which one to use. We can use the `sample_label` argument to indicate which method to use
during training.

- `all`: for one image that has x number of labels, the loader yields x image-label
  pairs with the same image. Occurs over all images, over one epoch.
- `sample`: for one image that has x number of labels, the loader yields 1 image-label
  pair randomly sampled from all the labels. Occurs for all images in one epoch.

During validation and testing (ie for `valid` and `test` directories), data loaders will
be built to sample `all` the data-label pairs, regardless of the argument passed to
`sample_label`.

```yaml
dataset:
  train:
    dir: "data/test/h5/paired/train"
    format: "h5"
    labeled: true
  valid:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  type: "paired" # one of "paired", "unpaired" or "grouped"
  sample_label: "sample" # one of "sample", "all" or None
```

In the case the `labeled` argument is false, the sample_label is unused, but still must
be passed. Additionally, if the tensors in the files only have one label, regardless of
the `sample_label` argument, the data loader will only pass the one label to the
network.

For more details please refer to
[Read The Docs](https://deepreg.readthedocs.io/en/latest/docs/exp_label_sampling.html).

#### Paired

- `moving_image_shape`: Union[Tuple[int, ...], List[int]] of ints, len 3, corresponding
  to (dim1, dim2, dim3) of the 3D moving image.
- `fixed_image_shape`: Union[Tuple[int, ...], List[int]] of ints, len 3, corresponding
  to (dim1, dim2, dim3) of the 3D fixed image.

```yaml
dataset:
  train:
    dir: "data/test/h5/paired/train"
    format: "h5"
    labeled: true
  valid:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  type: "paired" # one of "paired", "unpaired" or "grouped"
  sample_label: "sample" # one of "sample", "all" or None
  moving_image_shape: [16, 16, 3]
  fixed_image_shape: [16, 16, 3]
```

#### Unpaired

- `image_shape`: Union[Tuple[int, ...], List[int]] of ints, len 3, corresponding to
  (dim1, dim2, dim3) of the 3D image.

```yaml
dataset:
  train:
    dir: "data/test/h5/paired/train"
    format: "h5"
    labeled: true
  valid:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  type: "unpaired" # one of "paired", "unpaired" or "grouped"
  sample_label: "sample" # one of "sample", "all" or None
  image_shape: [16, 16, 3]
```

#### Grouped

- `intra_group_prob`: float, between 0 and 1. Passing 0 would only generate inter-group
  samples, and passing 1 would only generate intra-group samples.
- `sample_label`: method for sampling the labels "sample", "all".
- `intra_group_option`: str, "forward", "backward, or "unconstrained"
- `sample_image_in_group`: bool, if true, only one image pair will be yielded for each
  group, so one epoch has num_groups pairs of data, if false, iterate through this
  loader will generate all possible pairs.
- `image_shape`: Union[Tuple[int, ...], List[int]] len 3, corresponding to (dim1, dim2,
  dim3) of the 3D image.

```yaml
dataset:
  train:
    dir: "data/test/h5/paired/train"
    format: "h5"
    labeled: true
  valid:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  test:
    dir: "data/test/h5/paired/test"
    format: "h5"
    labeled: true
  type: "grouped" # one of "paired", "unpaired" or "grouped"
  sample_label: "sample" # one of "sample", "all" or None
  image_shape: [16, 16, 3]
  sample_image_in_group: true
  intra_group_prob: 0.7
  intra_group_option: "forward"
```

See the [dataset loader configuration](dataset_loader.html) for more details.

## Train section

The `train` section defines the neural network training hyper-parameters, by specifying
subsections, `method`, `backbone`, `loss`, `optimizer`, `preprocess` and other training
hyper-parameters, including `epochs` and `save_period`.

### Method - required

The `method` argument defines the registration type. It must be a string. Feasible
values are: `ddf`, `dvf`, and `conditional`, corresponding to the dense displacement
field (DDF) based model, dense velocity field (DDF) based model, and conditional model
presented in the [registration tutorial](../tutorial/registration.html).

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
```

### Backbone - required

The `backbone` subsection is used to define the network, with all the network-specific
arguments under the same indent. The first argument should be the argument `name`, which
should be string type, one of "unet", "local" or "global", to define a UNet, LocalNet or
GlobalNet backbone, respectively. With Registry functionalities, you can also define
your own networks to pass to DeepReg train via config.

The `num_channel_initial` is used to define the number of initial channels for the
network, and should be int type.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "unet" # One of unet, local, global: networks currently supported by DeepReg
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
```

#### UNet

The UNet model requires several additional arguments to define its structure:

- `depth`: int, defines the depth of the UNet from first to bottom, bottleneck layer.
- `pooling`: Boolean, pooling method used for down-sampling. True: non-parametrized
  pooling will be used, False: conv3d will be used.
- `concat_skip`: Boolean, concatenation method for skip layers in UNet. True:
  concatenation of layers, False: addition is used instead.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "unet" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    depth: 3
    pooling: false
    concat_skip: true
```

#### LocalNet

The LocalNet has an encoder-decoder structure and extracts information from tensors at
one or multiple resolution levels. We can define which levels to extract info from with
the `extract_levels` argument.

- `depth`: Depth of the encoder, `depth=2` means there are in total 3 layers where 0 is
  the top layer and 2 is the bottom.
- `extract_levels`: indices of layer from which the output will be extracted, the value
  range is `[0, depth]` both side inclusive.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    depth: 2
    extract_levels: [0, 1, 2]
```

#### GlobalNet

The GlobalNet has a U-net like encoder to encode the image and uses the bottleneck layer
to output an affine transformation using a CNN.

- `depth`: Depth of the encoder, `depth=2` means there are in total 3 layers where 0 is
  the top layer and 2 is the bottom.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "global" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    depth: 4
```

### Loss - required

This section defines the loss in training.

There are three different categories of losses in DeepReg:

- **image loss**: loss between the fixed image and predicted fixed image (warped moving
  image).
- **label loss**: loss between the fixed label and predicted fixed label (warped moving
  label).
- **regularization loss**: loss on predicted dense displacement field (DDF).

Not all losses are applicable for all models, the details are in the following table.

|                     | DDF / DVF                      | Conditional    |
| ------------------- | ------------------------------ | -------------- |
| Image Loss          | Applicable                     | Non-applicable |
| Label Loss          | Applicable if data are labeled | Applicable     |
| Regularization Loss | Applicable                     | Non-applicable |

The configuration for non-applicable losses will be ignored without errors. The loss
will also be ignored if the weight is zero. However, each model must define at least one
loss, otherwise error will be raised by TensorFlow.

For each loss, there are multiple existing loss functions to choose. The registry
mechanism can also be used to use custom loss functions. Please read the
[registry documentation](registry.html) for more details.

#### Image

The image loss calculates dissimilarity between warped image tensors and fixed image
tensors.

- `weight`: float type, the weight of individual loss element in the total loss
  function.
- `name`: string type, one of "lncc", "ssd" or "gmi".

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    image:
      name: "lncc" # other options include "lncc", "ssd" and "gmi", for local normalised cross correlation,
      weight: 0.1
```

The following are the DeepReg image losses. Additional arguments should be added at the
same indent level:

- `lncc`: Calls a local normalized cross-correlation type loss. Requires the following
  arguments:

  - `kernel_size`: int, optional, default=9. Kernel size or kernel sigma for
    kernel_type="gaussian".
  - `kernel_type`: str, optional, default="rectangular". One of "rectangular",
    "triangular" or "gaussian"

- `ssd`: Calls a sum of squared differences loss. No additional arguments required.

- `gmi`: Calls a global mutual informatin loss. Requires the following arguments:
  - `num_bins`: int, optional, default=23. Number of bins for intensity.
  - `sigma_ratio`: float, optional, default=0.5. A hyperparameter for the Gaussian
    kernel density estimation.

#### Label

The label loss calculates dissimilarity between labels.

All default DeepReg losses can be used as multi-scale or single scale losses.
Multi-scale losses require a kernel Additionally, all losses can be weighted, so the
following two arguments are global to all provided losses:

- `weight`: float type, weight of individual loss element in total loss function.
- `scales`: list of ints, or None. Optional argument. If you do not pass this argument
  (or pass the list [0], the value `null` or an empty value pair), the loss is
  calculated at a single scale. If you pass a list of length > 1, a multi-scale loss
  will be used. WARNING: an empty list ([]) will raise an error.
- `kernel`: str, "gaussian" or "cauchy", default "gaussian". Optional argument. Defines
  the kernel to use for multi-scale losses.

EG.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    label:
      weight: 1.0
      name: "dice" # options include "dice", "cross-entropy", "mean-squared", "generalised_dice" and "jaccard"
      scales: [1, 2]
```

The default losses require the following arguments. Additional arguments should be added
at the same indent level:

- `dice`: Calls a Dice loss on the labels, requires the following arguments:

  - `binary`: bool, default is false. If true, the tensors are thresholded at 0.5.
  - `background_weight`: float, default=0.0. `background_weight` weights the foreground
    and background classes by replacing the labels of 1s and 0s with
    `(1-background_weight)` and `background_weight`, respectively.

- `cross-entropy`: Calls a cross-entropy loss between labels, requires the following
  arguments:

  - `binary`: bool, default is false. If true, the tensors are thresholded at 0.5.
  - `background_weight`: float, default=0.0. `background_weight` weights the foreground
    and background classes by replacing the labels of 1s and 0s with
    `(1-background_weight)` and `background_weight`, respectively.

- `jaccard`: - `binary`: bool, default is false. If true, the tensors are thresholded at
0.5.
<!-- - `background_weight`: float, default=0.0. `background_weight` weights the foreground and background classes by replacing the labels of 1s and 0s with (1-background_weight) and background_weight, respectively. -->

#### Regularization

The regularization section configures the losses for the DDF. To instantiate this part
of the loss, pass "regularization" into the config file as a field.

- `weight`: float type, the weight of the regularization loss.
- `name`: string type, the type of deformation energy to compute. Options include
  "bending", "gradient"

If the `gradient` loss is used, another argument must be passed at the same indent
level: - `l1`: bool. Indicates whether to calculate the L1-norm (true) or L2-norm
(false) gradient loss of the ddf.

EG.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    regularization:
      weight: 0.5 # weight of regularization loss
      name: "bending" # options include "bending", "gradient"
```

or

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    regularization:
      weight: 0.5 # weight of regularization loss
      name: "gradient" # options include "bending", "gradient"
      l1: false
```

#### Composite Loss

The loss function can be a composite of different loss categories by adding all fields
in the same configuration file.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    image:
      name: "gmi"
      weight: 1.0
    label:
      weight: 1.0
      name: "dice"
```

Moreover, one may want to specify several loss functions for each category. In that
case, a dashed line (-) indicates the specification of a new loss function under each
field.

E.G.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    image:
      - name: "lncc"
        weight: 0.5
        kernel_size: 5
      - name: "gmi"
        weight: 0.5
    label:
      name: "dice"
      weight: 0.5
```

### Optimizer - required

The optimizer can be defined by using a `name` with other required arugment. The name
must be the same to the class name under `tf.keras.optimizers`.

For instance, to use a default
[Keras Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam),
the configuration should be

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    regularization:
      weight: 0.5 # weight of regularization loss
      name: "bending" # options include "bending", "gradient"
  optimizer:
    name: "Adam"
```

For a
[Keras SGD optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD)
with learning rate 0.001, the configuration should be

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    regularization:
      weight: 0.5 # weight of regularization loss
      name: "bending" # options include "bending", "gradient"
  optimizer:
    name: "SGD"
    learning_rate: 0.001
```

### Preprocess - required

The `preprocess` field defines how the data loader feeds data into the model.

- `batch_size`: int, specifies the number of samples per step for prediction. If using
  multiple GPUs, i.e. `n` GPUs, each GPU will have mini batch size `batch_size / n`.
  Thus, `batch_size` should be divided by `n` evenly.
- `shuffle_buffer_num_batch`: int, helps define how much data should be pre-loaded into
  memory to buffer training, such that shuffle_buffer_size = batch_size \*
  shuffle_buffer_num_batch.
- `num_parallel_calls`: int, it defines the number of cpus used during preprocessing, -1
  means unlimited and it may take all cpus and significantly more memory.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    regularization:
      weight: 0.5 # weight of regularization loss
      name: "bending" # options include "bending", "gradient"
  optimizer:
    name: "sgd"
    sgd:
      learning_rate: 1.0e-5
      momentum: 0.9
      nesterov: false
  preprocess:
    batch_size: 32
    shuffle_buffer_num_batch: 1
    num_parallel_calls: -1 # number elements to process asynchronously in parallel during preprocessing, -1 means unlimited, heuristically it should be set to the number of CPU cores available
```

### Epochs - required

The `epochs` field defines the number of epochs to train the network for.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    regularization:
      weight: 0.5 # weight of regularization loss
      name: "bending" # options include "bending", "gradient"
  optimizer:
    name: "sgd"
    sgd:
      learning_rate: 1.0e-5
      momentum: 0.9
      nesterov: false
  preprocess:
    batch_size: 32
    shuffle_buffer_num_batch: 1
  epochs: 1000
```

### Saving frequency - required

The `save_period` field defines the save frequency - the model will be saved every
`save_period` epochs.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
    regularization:
      weight: 0.5 # weight of regularization loss
      name: "bending" # options include "bending", "gradient"
  optimizer:
    name: "sgd"
    sgd:
      learning_rate: 1.0e-5
      momentum: 0.9
      nesterov: false
  preprocess:
    batch_size: 32
    shuffle_buffer_num_batch: 1
  epochs: 1000
  save_period: 5
```
