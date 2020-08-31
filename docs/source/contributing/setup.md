# Set Up

To edit the source code of DeepReg, besides the
[package installation](../getting_started/install.html), we recommend installing
[pre-commit](https://pre-commit.com/) for code style consistency and auto formatting
before each commit to prevent unnecessary linting failure in
[Travis-CI](https://travis-ci.org/github/DeepRegNet/DeepReg).

## Pre-commit

We are currently using (by order) the following pre-commit hooks:

- [seed-isort-config](https://github.com/asottile/seed-isort-config) and
  [isort](https://github.com/timothycrosley/isort) to format package imports in python
  files.
- [Black](https://github.com/psf/black) to format python files.
- [Flake8](https://gitlab.com/pycqa/flake8) to perform python linting check,
- [Prettier](https://prettier.io/) to format markdown files.

### Installation

Pre-commit is installed during the package installation via `pip install -e .`. To
activate pre-commit, make sure the git is installed (`sudo apt install git` for linux)
and run `pre-commit install` under the root of this repository `DeepReg/`.

### Usage

We can use `pre-commit run --all-files` to trigger the hooks manually to format all
files before pull request.

Optionally, we can use
`git commit --no-verify -m "This is a commit message placeholder."` to skip pre-commit.
However, this is not recommended.

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
please raise an issue.
