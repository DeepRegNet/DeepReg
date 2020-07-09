# Contributing to DeepReg

We welcome contributions to DeepReg.

## Reporting bugs and feature requests

Please create a new issue on: https://github.com/ucl-candi/DeepReg/issues/new

When reporting a bug, please include:

1. The version of DeepReg you are using
2. Your OS version (for example Windows 10 64-bit, macOS High Sierra, Ubuntu 16.04)
3. Detailed steps to reproduce the bug.

## Fixing bugs or implement features

The easiest way to contribute is to follow these guidelines:

1. Look through the issues on https://github.com/ucl-candi/DeepReg/issues and assign the
   relevant issue to yourself. If there is not an existing issue that covers your work,
   please create one: https://github.com/ucl-candi/DeepReg/issues/new
2. Read the design considerations below.
3. Fork the repository: https://github.com/ucl-candi/DeepReg/forks/new
4. Create a branch for your changes. The branch name should start with the issue number,
   followed by hyphen separated words describing the issue. For example:
   1-update-contribution-guidelines
5. Make your changes following the coding guidelines below.
6. Commit and push your changes to your fork. The commit message should start with
   `Issue #<issue number>`, for example: "Issue #1: Fixed typo". Commit in small,
   related chunks. Review each commit and explain its purpose in the commit message.
   Refer to the commit style section below for a more detailed guide.
7. Submit a merge request: https://github.com/ucl-candi/DeepReg/merge-requests/new
8. Merge request will be reviewed and, if necessary, changes suggested before merge to
   master.

## Commit style

To facilitate review of contributions, we encourage contributors to adhere to the
following commit guidelines.

1. The commit message should start with `Issue #<issue number>` so that the commits are
   tied to a specific issue.
   - **Good**:
     `Issue #<issue number>: modified resample function docstring to reflect changes in function args`
   - **Moderate/OK**:
     `ref #<issue number>: modified resample function docstring to reflect changes in function args` -
     points to appropriate ticket, but not in described style.
   - **Bad**:
     `modified resample function docstring to reflect changes in function args` - we
     don't know what ticket this refers to!
2. Include related changes in the same commit, where possible, For example, if multiple
   typos are spotted in a document, aim to resolve them in the same commit instead of
   separate commits. This improves history readibility.
   - **Good**: One commit for the same type of problem
     `Issue #<issue number>: fixed typos in function x`
   - **Bad**: Multiple of `Issue #<issue number>: fixed typo in function x` in the same
     thread. Clutters repo history.
3. Strive to add informative commit messages.
   - **Good**:
     `Issue #<issue number>: removed unused arguments across function x in loops to comply with PEP8 standard in file y`,
     `Issue #<issue number>: added new data loader class inheriting from base class to deal with np array file format`
   - **Not acceptable**: `Issue #<issue number>: lint`,
     `Issue #<issue number>: add loader` - hard to tell explicitly what was changed
     without doing an in depth review.

## Design considerations

1. As few dependencies as possible. Try to stick to standard scipy packages like numpy
   and pandas.
2. Discuss extra dependencies with the team and maybe the outcome will be to create a
   new separate package, where you can be more specific and more modular.
3. Unit test well, using pytest, with good coverage.
4. All errors as exceptions rather than return codes.

## Coding guidelines

1. Please follow PEP8 guidelines https://www.python.org/dev/peps/pep-0008/
2. Create a python virtual environment (virtualenv) for development
3. Make sure that pylint passes. You may disable specific warnings within the code where
   it is reasonable to do so
4. Add unit tests for new and modified code
5. Make sure all existing and new tests pass
6. Make sure all docstrings have been added
7. Make sure all dependencies have been added to requirements
8. Make sure your code works for all required versions of Python
9. Make sure your code works for all required operating systems
10. CI is enabled: for documentation changes/linting commits you may include [ci skip]
    in your commit messages. A reminder that CI checks are required before merge!

## Pre-commit setup

[pre-commit](https://pre-commit.com/) is recommended for code style consistency before
each commit. Specifically,

- [seed-isort-config](https://github.com/asottile/seed-isort-config) and
  [isort](https://github.com/timothycrosley/isort) to format package imports in python
  files.
- [Black](https://github.com/psf/black) to format python files.
- [Flake9](https://gitlab.com/pycqa/flake8) to perform linting check,
- [Prettier](https://prettier.io/) to format markdown files.

Pre-commit is installed during the package installation via `pip install -e .`. To
activate pre-commit, make sure the git is installed (`sudo apt install git` for linux)
and run `pre-commit install` under the root of this repository `DeepReg/`.

Optionally, use `git commit --no-verify -m "This is a commit message placeholder."` to
skip pre-commit, and use `pre-commit run --all-files` to format files before pull
request.

## Documentation Pages

[Docsify](https://docsify.js.org/) converts markdown files into pages and they are
hosted on github page. Use `cd docs && python -m SimpleHTTPServer 3000` to visualize the
pages locally. The required package
[simple_http_server](https://github.com/keijack/python-simple-http-server) has been
installed during the package installation via `pip install -e .`.
