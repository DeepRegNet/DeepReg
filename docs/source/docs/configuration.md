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

`seed`:

##### Paired

`moving_image_shape`: (list, tuple) `fixed_image_shape`: (list, tuple)

##### Unpaired

`image_shape`:

##### Grouped

`intra_group_prob`: float, between 0 and 1. `sample_label`:method for sampling the
labels "sample" "first" "all" `intra_group_option`: str, "forward", "backward, or
"unconstrained" `sample_image_in_group`: bool, `image_shape`

See the [dataset loader configuration](dataset_loader.html) for more details.

## Train section

The `train` section defines the neural network training hyper-parameters, by specifying
subsections, `model`, `loss`, `optimizer`, `preprocess` and other training
hyper-parameter, including `epochs` and `save_period`. See an
[example configuration](https://github.com/DeepRegNet/DeepReg/blob/main/config/unpaired_labeled_ddf.yaml),
with comments on the available options in each subsection.

This section is highly application-specific. More examples can be found in
[DeepReg Demos](../demo/introduction.html).
