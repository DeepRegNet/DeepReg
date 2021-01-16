# Change Log

All notable changes to this project will be documented in this file. It's a team effort
to make them as straightforward as possible.

The format is based on [Keep a Changelog](http://keepachangelog.com/) and this project
adheres to [Semantic Versioning](http://semver.org/).

## [0.1.1] - In Progress

### Added

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
