# Demo Requirement

v0.7

The [demos](https://github.com/DeepRegNet/DeepReg/tree/master/demos) folder directly
under the DeepReg root directory contains demonstrations using DeepReg for different
image registration applications.

Contributions are welcome! Below is a set of requirements for a demo to be included as a
DeepReg Demo.

- Each demo _must_ have an independent folder directly under the 'Demos';
- Name the folder as
  `[loader-type]_[image-modality]_[organ-disease]_[optional:brief-remark]`, e.g.
  `unpaired_ultrasound_prostate` or `grouped_mr_brain_logitudinal`;
- For simplicity, avoid sub-folders (other than those specified below) and separate
  files for additional functions/classes;
- Experiment using cross-validation or advanced data set sampling is NOT encouraged,
  unless the purpose of the demo is to demonstrate how to design experiments.

## Open accessible data

- Each demo _must_ have a 'demo_data.py' script to automatically download demo data;
- Data should be downloaded under the demo folder named `dataset`;
- Data should be hosted in a reliable and efficient (DeepReg repo will not store demo
  data or model) online storage, Kaggle, GitHub and Zendoo are all options for non-login
  access (avoid google drive for known accessibility issues);
- Relevant dataset folder structure to utilise the supported loaders can be either
  pre-arranged in data source or scripted in 'demo_data.py' after downloading;
- Avoid slow and excessively large data set download. Consider downloading a subset as
  default for demonstration purpose, with options for full data set.

## Pre-trained model

- A pre-trained model _must_ be available for downloading, with
  github.com/DeepRegNet/deepreg-model-zoo being preferred for the storing the models.
  Please contact the Dev Team for access;
- The pre-trained model, e.g. ckpt files, should be downloaded and extracted under the
  `dataset` folder. Avoid overwriting with user-trained models;

## Training

- Each demo _must_ have a 'demo_train.py' script. If using command line interface
  `deepreg_train`, this file needs to print a message to direct the user to the
  readme.md file (described below) for instructions;
- This is accompanied by one or more config yaml files in the same folder. If
  appropriate, please use the same demo folder name for the config file. Add postfix if
  using multiple config files, e.g. `unpaired_lung_ct_dataset.yaml`,
  `unpaired_lung_ct_train.yaml`.

## Predicting

- Each demo _must_ have a 'demo_predict.py' script; If using command line interface
  `deepreg_predict`, this file needs print a message to direct the user to the readme.md
  file (described below) for instructions;
- By default, the pre-trained model should be used in 'demo_predict.py'. However, the
  instruction should be clearly given to use the user-trained model, saved with the
  'demo_train.py';
- Report registration results. Provide at least one piece of numerical metric (Dice,
  distance error, etc) to show the efficacy of the registration. Optimum performance is
  not required;
- Provide at least one piece of visualisation of the results, such as moving image vs
  fixed image vs warped moving image (pred_fixed_image). This may be simply done by
  selecting the typical results from the predict output. If possible, save the
  visualisation to (e.g. png/jpg) files, avoiding compatibility issues.

## A 'readme.md' file

The markdown file _must_ be provided as an entry point for each demo, which should
contain the following sections:

- [Demo name] - Use the first-level heading with # and all the following are using the
  second-level subheadings with ##;
- [Author] Author name and email;
- [Application] Briefly describe the clinical application and the need for registration;
- [Instruction] A step-by-step instruction how the demo can be run. Preferably, use the
  demo folder as working directory;
- [Pre-trained Model] Clearly indicate that the demo uses pre-trained model by default,
  while give clear instruction how the alternative user-trained model can be used;
- [Data] Acknowledge data source;
- [Tested DeepReg Version] Demos do not need to be unit-tested. Record the commit
  \#hashtag on which it is tested.

## Code style

- Please restrict using external libraries or anything unsupported by
  [Colab](colab.research.google.com) or [Azure](https://notebooks.azure.com/);
- See general
  [Contribution Guide](https://github.com/ucl-candi/DeepReg/blob/master/docs/CONTRIBUTING.md).
