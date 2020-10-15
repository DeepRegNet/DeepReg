---
name: 🐜 Bug report
about: Something isn't working.
---

## Subject of the issue

Describe your issue here.

If the bug is confirmed, would you be willing to submit a PR? _(Help can be provided if
you need assistance submitting a PR)_

Yes / No

## Your environment

- DeepReg version (commit hash)

  Please use `git rev-parse HEAD` to get the hash of the current commit. Using
  `pip list` will provide the fixed tag version inside `setup.py`, therefore it is not
  accurate.

  We recommend installing DeepReg using
  [Anaconda](https://docs.anaconda.com/anaconda/install/) /
  [Miniconda](https://docs.conda.io/en/latest/miniconda.html) in a separate virtual
  environment.

- OS (e.g. Ubuntu 20.04, MacOS 10.15, etc.)

  We do not officially support windows.

- Python Version (3.7, 3.8, etc.)

  We support only Python 3.7 officially.

- TensorFlow

  - TensorFlow Version (2.2, 2.3, etc.)
  - CUDA Version (10.1, etc.) if available.
  - cuDNN Version

  We support only TensorFlow 2.3 officially.

  If using GPU, please check https://www.tensorflow.org/install/source#gpu to verify the
  GPU support.

## Steps to reproduce

Tell us how to reproduce this issue. Please provide a working demo.

## Expected behaviour

Tell us what should happen.

## Actual behaviour

Tell us what happens instead.
