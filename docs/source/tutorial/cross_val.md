# Design Experiments

DeepReg dataset loaders use a folder/directory-based file storing approach, with which
the user will be responsible for
[organising image and label files in required file formats and folders](../docs/dataset_loader.html).
This design was primarily motivated by the need to minimise the risk of data leakage (or
information leakage), both in code development and subsequent applications.

## Random-split

Every call of the `deepreg_train` or `deepreg_predict` function uses a dataset
"physically" separated by folders, including 'train', 'val' and 'test' sets used in a
random-split experiment. In this case, the user needs to randomly assign available
experiment image and label files into the three folders. Again, for more details see the
[Dataset loader](../docs/dataset_loader.html).

## Cross-validation

Experiments such as _cross-validation_ can be readily implemented by using the
"multi-folder support" in the `dataset` section of the yaml configuration files. See
details in [configuration](../docs/configuration.html).

For example, in a 3-fold cross-validation, the user may randomly partition available
experiment data files into four folders, 'fold0', 'fold1', 'fold2' and 'test'. The
'test' is a hold-out testing set. Each run of the 3-fold cross-validation then can be
specified in a different yaml file as follows.

"cv_run1.yaml":

```yaml
dataset:
  dir:
    train: # training data set
      - "data/test/h5/paired/fold0"
      - "data/test/h5/paired/fold1"
    valid: "data/test/h5/paired/fold2" # validation data set
    test: ""
```

"cv_run2.yaml":

```yaml
dataset:
  dir:
    train: # training data set
      - "data/test/h5/paired/fold0"
      - "data/test/h5/paired/fold2"
    valid: "data/test/h5/paired/fold1" # validation data set
    test: ""
```

"cv_run3.yaml":

```yaml
dataset:
  dir:
    train: # training data set
      - "data/test/h5/paired/fold1"
      - "data/test/h5/paired/fold2"
    valid: "data/test/h5/paired/fold0" # validation data set
    test: ""
```

To further facilitate flexible uses of these dataset loaders, the `deepreg_train` and
`deepreg_predict` functions also accept multiple yaml files - therefore the same `train`
section does not have to be repeated multiple times for the multiple cross-validation
folds or for the test. An example `dataset` section for configuring testing when using
`deepreg_predict` is given below.

"test.yaml":

```yaml
dataset:
  dir:
    train: ""
    valid: ""
    test: "data/test/h5/paired/test" # validation data set
```
