# Guidelines

We welcome contributions to DeepReg. Please
[raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose) to report
bugs, request features, or ask questions. For code contribution, please follow the
guidelines below.

## Setup

We recommend using conda environment on Linux or Mac machines for code development. The
setup steps are:

1. [Install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
2. [Clone or fork](https://github.com/DeepRegNet/DeepReg) the repository.
3. [Install](../getting_started/install.html) and activate `deepreg` conda environment.
4. Run `pre-commit install` under the root of this repository `DeepReg/` to install
   pre-commit hooks.

## Resolve an issue

For resolving an issue, please

1. Create a branch.

   The branch name should start with the issue number, followed by hyphen separated
   words describing the issue, e.g. `1-update-contribution-guidelines`.

2. Implement the required features or fix the reported bugs.

   There are several guidelines for commit, coding, testing, and documentation.

   1. Please create commits with meaningful commit messages.

      The commit message should start with `Issue #<issue number>:`, for instance
      `Issue #1: add commit requirements.`

   2. Please write or update unit-tests for the added or changed functionalities.

      Pull request will not be approved if test coverage is decreased. Check
      [testing guidelines](test.html) for further details.

   3. Please write meaningful docstring and documentations for added or changed code.

      We use
      [Sphinx docstring format](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html).

   4. Please update the
      [CHANGELOG](https://github.com/DeepRegNet/DeepReg/blob/main/CHANGELOG.md)
      regarding the changes.

3. [Create a pull request](https://github.com/DeepRegNet/DeepReg/pulls) when the branch
   is ready.

   Please resume the changes with some more details in the description and check the
   boxes after submitting the pull request. Optionally, you can create a pull request
   and add `WIP` in the name to mark is as working in progress.

## Add a DeepReg demo

Adding DeepReg demo should be done via pull request. Besides the guidelines above,
adding demo has additional requirements described in [demo guidelines](demo.html).
