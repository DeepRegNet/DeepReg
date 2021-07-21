# Change Log

All notable changes to this project will be documented in this file. It's a team effort
to make them as straightforward as possible.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project
adheres to [Semantic Versioning](http://semver.org/).

## [1.0.0-rc1] - In Progress

Release comment: refactoring of models means that old checkpoint files are no longer
compatible with the updates.

### Added

- Added `num_parallel_calls` option in config for data preprocessing.
- Added tests for Dice score, Jaccard Index, and cross entropy losses.
- Added statistics on inputs, DDF and TRE into tensorboard.
- Added example for using custom loss.
- Added tests on Mac OS.
- Added tests for python 3.6 and 3.7.
- Added support to custom layer channels in U-Net.
- Added support to multiple loss functions for each loss type: "image", "label" and
  "regularization".
- Added LNCC computation using separable 1-D filters for all kernels available

### Changed

- Updated pre-trained models for unpaired_ct_abdomen demo to new version
- Changed dataset config so that `format` and `labeled` are defined per split.
- Reduced TensorFlow logging level.
- Used `DEEPREG_LOG_LEVEL` to control logging in DeepReg.
- Increased all EPS to 1e-5.
- Clarify the suggestion in doc to use all-zero masks for missing labels.
- Moved contributor list to a separate page.
- Changed `no-test` flag to `full` for demo scripts.
- Renamed `neg_weight` to `background_weight`.
- Renamed `log_dir` to `exp_name` and `log_root` to `log_dir` respectively.
- Uniformed local-net, global-net, u-net under a single u-net structure.
- Simplified custom layer definitions.
- Removed multiple unnecessary custom layers and use tf.keras.layers whenever possible.
- Refactored BSplines interpolation independently of the backbone network and available
  only for DDF and DVF models.

### Fixed

- Fixed using GPU remotely
- Fixed LNCC loss regarding INF values.
- Removed loss weight checks to be more robust.
- Fixed import error under python 3.6.
- Fixed the residual module in local net architecture, compatible for previous
  checkpoints.
- Broken link in README to seminar video.

## [0.1.2] - 2021-01-31

Release comment: This is mainly a bugfix release, although some of the tasks in
1.0.0-rc1 have been included in this release, with or without public-facing
accessibility (see details below).

### Added

- Added global NCC loss
- Added the docs on registry for backbone models.
- Added backward compatible config parser.
- Added tests so that test coverage is 100%.
- Added config file docs with details on how new config works.
- Added DDF data augmentation.
- Added the registry for backbone models and losses.
- Added pylint with partial check (C0103,C0301,R1725,W0107,W9012,W9015) to CI.
- Added badges for code quality and maintainability.
- Added additional links (CoC, PyPI) and information (contributing, citing) to project
  README.md.
- Added CMIC seminar where DeepReg was introduced to the project README.md.
- Added deepreg_download entry point to access non-release folders required for Quick
  Start.

### Changed

- Refactored optimizer configuration.
- Refactored affine transform data augmentation.
- Modified the implementation of resampler to support zero boundary condition.
- Refactored loss functions into classes.
- Use CheckpointManager callback for saving and support training restore.
- Changed distribute strategy to default for <= 1 GPU.
- Migrated from Travis-CI to GitHub Actions.
- Simplified configuration for backbone models and losses.
- Simplified contributing documentation.
- Uniform kernel size for LNCC loss.
- Improved demo configurations with the updated pre-trained models for:
  grouped_mask_prostate_longitudinal, paried_mrus_prostate, unpaired_us_prostate_cv,
  grouped_mr_heart, unpaired_ct_lung, paired_ct_lung.

### Fixed

- Fixed several dead links in the documentation.
- Fixed a bug due to typo when image loss weight is zero, label loss is not applied.
- Fixed warp CLI tool by saving outputs in Nifti1 format.
- Fixed optimiser storage and loading from checkpoints.
- Fixed bias initialization for theta in GlobalNet.
- Removed invalid `first` argument in DataLoader for sample_index generator.
- Fixed build error when downloading data from the private repository.
- Fixed the typo for CLI tools in documents.

## [0.1.0] - 2020-11-02

### Added

- Added option to change the kernel size and type for LNCC image similarity loss.
- Added visualization tool for generating gifs from model outputs.
- Added the max_epochs argument for training to overwrite configuration.
- Added the log_root argument for training and prediction to customize the log file
  location.
- Added more meaningful error messages for data loading.
- Added integration tests for all demos.
- Added environment.yml file for Conda environment creation.
- Added Dockerfile.
- Added the documentation about using UCL cluster with DeepReg.

### Changed

- Updated TensorFlow version to 2.3.1.
- Updated the pre-trained models in MR brain demo.
- Updated instruction on Conda environment creation.
- Updated the documentation regarding pre-commit and unit-testing.
- Updated the issue and pull-request templates.
- Updated the instructions for all demos.
- Updated pre-commit hooks version.
- Updated JOSS paper to address reviewers' comments.
- Migrated from travis-ci.org to travis-ci.com.

### Fixed

- Fixed prediction error when number of samples cannot be divided by batch size exactly.
- Fixed division by zero handling in multiple image/label losses.
- Fixed tensor comparison in unit tests and impacted tests.
- Removed normalization of DDF/DVF when saving in Nifti formats.
- Fixed invalid link in the quick start page.

## [0.1.0b1] - 2020-09-01

Initial beta release.
