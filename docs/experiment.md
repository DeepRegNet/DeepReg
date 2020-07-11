# Experiment design

`DeepReg` dataset loaders use a folder/directory-based file storing approach, which the
user will be responsible to
[organise their image and label files in the required file formats and folders](doc_data_loader.md).
This design was primarily motivated by the need to minimise the risk of data leakage (or
information leakage), both in code development and subsequent applications.

## Random-split

Every call of the function `train` or `predict` uses a dataset "physically" seperated by
folders, such as 'train', 'val' and 'test' sets used in a random-split experiment. In
this case, user needs to randomly assign available experiment image and label files into
the three folders. Again, for more details see the [Dataset loader](doc_data_loader.md).

## Cross-validation

Experiments such as _cross-validation_ can be implemented by using the "multi-folder
support" in the `dataset` section of the yaml configuration files. See more details in
[configuration](doc_configurtion.md).

For example, in a 3-fold cross-validation, user may randomly partition available
experiment data files into four folders, 'fold0', 'fold1', 'fold2' and 'test', with the
last of which being a hold-out testing set. Each run of the the 3-fold cross-validation
then can be specified in three different yaml files as follows.

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

The `train` and `predict` also accepts multiple yaml files to overwrite the previous
configurtions, therefore the `train` section of the yaml configuration files need not to
be repeated multiple times for the multiple cross-validation folds or for the test. An
example additional `dataset` section of the yaml configuration file for testing using
`predict` is given below.

"test.yaml":

```yaml
dataset:
  dir:
    train: ""
    valid: ""
    test: "data/test/h5/paired/test" # validation data set
```
