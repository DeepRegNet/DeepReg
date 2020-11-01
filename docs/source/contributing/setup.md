# Set Up

To edit the source code of DeepReg, besides the
[package installation](../getting_started/install.html), we recommend installing
[pre-commit](https://pre-commit.com/) for code style consistency and auto formatting
before each commit to prevent unnecessary linting failure in
[Travis-CI](https://travis-ci.com/github/DeepRegNet/DeepReg).

## Pre-commit

### Installation

Before installing `pre-commit`, please make sure the git is installed
(`sudo apt install git` for linux). Then please execute `pre-commit install` under the
root of this repository `DeepReg/` to install `pre-commit`.

### Usage

The `pre-commit` hooks will be activated automatically. But in case some files are not
properly formatted, please execute `pre-commit run --all-files` manually to format all
files.

### Linting conflicts

Sometimes, Black might have conflicts with flake8 and below are some possible cases and
work around.

- If a code is followed by a long comment in the same line, Black attempts to break
  lines. So we should put comment in the line above instead.
- For lists/tuples, do not add comma after the last element, unless it's a single
  element tuple, like `(1,)`.

To check if Black is causing conflicts, run `black .` in the root of DeepReg you will
see the formatted files by Black, run `pre-commit run --all-files`, you will see the
final versions. Compare them to understand an issue. If there's a new conflict case,
please [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose).
