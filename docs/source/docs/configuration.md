# Configuration File

Besides the arguments provided to the command line tools, detailed training and
prediction configuration is specified in a `yaml` file. The configuration file contains
two sections, `dataset` and `train`. Within `dataset` one specifies the data types,
sizes, as well as the data loader to use. The `train` section specifies parameters
related to the neural network. `yaml` files are used as they can be easily generated
from basic Python dictionary structures.

## Dataset section

The `dataset` section specifies the path to the data to be used during training, the
data loader to use as well as the specific arguments to configure the data loader.

### Dir key - Required

The paths to the training, validation and testing data are specified under a `dir`
dictionary key like this:

```yaml
dataset:
  dir:
    train: "data/test/h5/paired/train" # folder contains training data
    valid: "data/test/h5/paired/valid" # folder contains validation data
    test: "data/test/h5/paired/test" # folder contains test data
```

Multiple dataset directories can be specified, such that data is sampled across several
folders:

```yaml
dataset:
  dir:
    train:
      - "data/test/h5/paired/train1"
      - "data/test/h5/paired/train2" # folders contains training data
    valid: "data/test/h5/paired/valid" # folder contains validation data
    test: "data/test/h5/paired/test" # folder contains test data
```

### Format key - Required

The type of data we supply the data loaders will influence the behaviour, so we must
specify the data type using the `format` key. Currently, DeepReg data loaders support
nifti and h5 file types - alternate file formats will raise errors in the data loaders.
To indicate which format to use, pass a string to this field as either "nifti" or "h5":

```yaml
dataset:
  dir:
    train: "data/test/h5/paired/train1" # folders contains training data
    valid: "data/test/h5/paired/valid" # folder contains validation data
    test: "data/test/h5/paired/test" # folder contains test data
  format: "nifti"
```

Depending on the data type, DeepReg will expect the images and labels to be stored in
specific structures: check the [data loader configuration](dataset_loader.html) for more
details.

### Labeled key - Required

The `labeled` key indicates whether labels should be used during training. A Boolean is
used to indicate the usage of labels:

```yaml
dataset:
  dir:
    train: "data/test/h5/paired/train1" # folders contains training data
    valid: "data/test/h5/paired/valid" # folder contains validation data
    test: "data/test/h5/paired/test" # folder contains test data
  format: "nifti"
  labeled: true
```

### Type key - Required

The type of data loader used will depend on how one wants to train the network.
Currently, DeepReg data loaders support the "paired", "unpaired" and "grouped" training
strategies. Passing a string that doesn't match any of the above would raise an error.
The data loader type would be specified using the `type` key:

```yaml
dataset:
  dir:
    train: "data/test/h5/paired/train1" # folders contains training data
    valid: "data/test/h5/paired/valid" # folder contains validation data
    test: "data/test/h5/paired/test" # folder contains test data
  format: "nifti"
  type: "paired" # one of "paired", "unpaired" or "grouped"
```

#### Data loader dependent keys

Depending on which string is passed to the `type` key, DeepReg will initialise a
different data loader instance with different sampling strategies. These are reviewed in
depth in the [dataset loader configuration](dataset_loader.html) documentation. Here we
outline the arguments necessary to configure the different dataloaders.

###### Sample_label - Required

In the case that we have more than one label per image, we need to indicate to the
loader which one to use. We can use the `sample_label` argument to indicate which method
to use. Pass one of "sample", for random sampling, or "all" to use all the available
labels:

```yaml
dataset:
  dir:
    train: "data/test/h5/paired/train1" # folders contains training data
    valid: "data/test/h5/paired/valid" # folder contains validation data
    test: "data/test/h5/paired/test" # folder contains test data
  format: "nifti"
  type: "paired" # one of "paired", "unpaired" or "grouped"
  labeled: true
  sample_label: "sample" # one of "sample", "all" or None
```

In the case the `labeled` argument is false, the sample_label is unused, but still must
be passed. Additionally, if the tensors in the files only have one label, irregardles of
the `sample_label` argument, the data loader will only pass the one label to the
network.

##### Paired

- `moving_image_shape`: (list, tuple)
- `fixed_image_shape`: (list, tuple)

##### Unpaired

`image_shape`:

##### Grouped

- `intra_group_prob`: float, between 0 and 1.
- `sample_label`: method for sampling the labels "sample" "first" "all"
  `intra_group_option`: str, "forward", "backward, or "unconstrained"
- `sample_image_in_group`: bool,
- `image_shape`

See the [dataset loader configuration](dataset_loader.html) for more details.

## Train section

The `train` section defines the neural network training hyper-parameters, by specifying
subsections, `method`, `backbone`, `loss`, `optimizer`, `preprocess` and other training
hyper-parameters, including `epochs` and `save_period`.

### Method - required

The `method` argument defines the registration type. It must be a string type, one of
"ddf", "dvf", "conditional", which are the currently supported registration methods.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
```

### Backbone - required

The `backbone` section defines the backbone network for the registration. This section
has several subsections. The `name` and `num_channel_initial` are global to all backbone
methods, and there are specific arguments for some of the backbones to define their
implementation.

#### Global parameters for train The `name` is used to define the network. It should be
string type, one of "unet", "local" or "global", to define a UNet, LocalNet or GlobalNet
backbone, respectively.

The `num_channel_initial` is used to define the number of initial channels for the
network, and should be int type.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "unet" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
```

#### UNet

The UNet model requires several additional arguments to define it's structure:

- `depth`: int, defines the depth of the UNet from first to bottom, bottleneck layer.
- `pooling`: Boolean, pooling method used for downsampling. True: non-parametrized
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
    pooling: False
    concat_skip: True
```

#### Local and GlobalNet The LocalNet has an encoder-decoder structure, and extracts
information from tensors at certain levels. We can define which levels to extract info
from with the `extract_levels` argument.

The GlobalNet encodes the image and uses the bottleneck layer to output an affine
transformation using a FCN.

- `extract_levels`: list of positive ints (ie, the min value in `extract_levels` should
  be >=0). Eg. [1, 2, 3, 5] will extract information at those levels (but not 4).

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
```

### Loss - required

This section defines the loss in training.

### Optimizer - required The optimizer can be defined by using a `name` and then passing
optimizer specific arguments with the same name. All optimizers can use the
`learning_rate` argument.

- `name`: string type, is used to define the optimizer during training. One of "adam",
  "sgd", "rms". There must be another additional field with the same name.

- `adam`: If adam is passed into `name`, the `adam` field must be passed. The dictionary
  can be empty, which initalises a default
  [Keras Adam optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam).
  Alternatively, fields with names equivalent to those specified in the optimizer
  documentation can be used.
- `sgd`: If sgd is passed into `name`, the `sgd` field must be passed. The dictionary
  can be empty, which initalises a default
  [Keras SGD optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD).
  Alternatively, fields with names equivalent to those specified in the optimizer
  documentation can be used instead.
- `rms`: If rms is passed into `name`, the `rms` field must be passed. The dictionary
  can be empty, which initalises a default
  [Keras RMSprop optimizer](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/RMSprop).
  Alternatively, fields with names equivalent to those specified in the optimizer
  documentation can be used instead.

```yaml
train:
  method: "ddf" # One of ddf, dvf, conditional
  backbone:
    name: "local" # One of unet, local, global
    num_channel_initial: 16 # Int type, number of initial channels in the network. Controls the network size.
    extract_levels: [0, 1, 2]
  loss:
  optimizer:
    name: "adam"
    adam:
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
  optimizer:
    name: "sgd"
    sgd:
      learning_rate: 1.0e-5
      momentum: 0.9
      nesterov: False
```

### Preprocess - required

- batch_size
- shuffle_buffer_num_batch

### Epochs - required

### save_period
