# Coding Requirement

To ensure the code quality and the consistency, we recommend the following guidelines.

## Coding design

1. Please use as few dependencies as possible. Try to stick to standard `scipy` packages
   like `numpy` and `pandas`.

   In case of adding new dependency, please make sure all dependencies have been added
   to requirements.

1. Please prevent adding redundant code. Try wrapping the code block into a function or
   class instead.

   In case of adding new functions, please make sure the corresponding unit tests are
   added.

1. When adding/modifying functions/classes, please make sure the corresponding
   docstrings are added/updated.

1. Please check [Unit Test Requirement](test.html) for detailed requirements on unit
   testing.

1. Please check [Documentation Requirement](docs.html) for detailed requirements on
   documentation,

## Continuous Integration (CI)

We use [Travis-CI](https://travis-ci.org/) for automated checking on

- linting
- unit test
- test coverage using [Codecov](https://codecov.io)
- documentation using [ReadTheDocs](https://readthedocs.org/).

For documentation changes/linting commits you may include `[ci skip]` in your commit
messages in order to skip the CI. However, CI checks are required before merging!

## Pre-commit

We recommend [pre-commit](https://pre-commit.com/) for code style consistency and auto
formatting before each commit to prevent unnecessary linting failure in CI.

Specifically, we are using (by order) the following pre-commit hooks:

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

### Common errors

Sometimes, black might have conflicts with prettier or flake8 and below are some
possible cases and work around.

- If a code is followed by a long comment in the same line, Black attempts to break
  lines. So we should put comment in the line above instead.
- For lists/tuples, do not add comma after the last element, unless it's a single
  element tuple, like `(1,)`.

To check if Black is causing conflicts, run `black .` in the root of DeepReg you will
see the formatted files by Black, run `pre-commit run --all-files`, you will see the
final versions. Compare them to understand an issue. If there's a new conflict case,
please raise an issue.
