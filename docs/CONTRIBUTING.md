# Contributing to DeepReg

We welcome contributions to DeepReg.

## Reporting bugs and feature requests

Found a bug? Create a [new issue](https://github.com/DeepRegNet/DeepReg/issues/new)!

When reporting a bug, please include:

1. The version of DeepReg you are using
2. Your OS version (for example Windows 10 64-bit, macOS High Sierra, Ubuntu 16.04)
3. Detailed steps to reproduce the bug.

## Fixing bugs or implement features

The easiest way to contribute is to follow these guidelines:

1. Look through the [issues](https://github.com/DeepRegNet/DeepReg/issues) and assign
   the relevant issue to yourself. If there is not an existing issue that covers your
   work, please create [one](https://github.com/DeepRegNet/DeepReg/issues/new).
2. Read the design considerations below.
3. [Fork the repository](https://github.com/DeepRegNet/DeepReg/forks/new)
4. Create a branch for your changes. The branch name should start with the issue number,
   followed by hyphen separated words describing the issue.
   - Example: `1-update-contribution-guidelines`
5. Make your changes following the coding guidelines below.
6. Commit and push your changes to your fork. The commit message should start with
   `Issue #<issue number>`, for example: "Issue #1: Fixed typo". Commit in small,
   related chunks. Review each commit and explain its purpose in the commit message.
   Refer to the commit style section below for a more detailed guide.
7. [Submit a pull request](https://github.com/DeepRegNet/DeepReg/merge-requests/new)
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

1. As few dependencies as possible. Try to stick to standard `scipy` packages like
   `numpy` and `pandas`.
2. Discuss extra dependencies with the team and maybe the outcome will be to create a
   new separate package, where you can be more specific and more modular.
3. Unit test well, using `pytest`, with good coverage.
4. All errors as exceptions rather than return codes.

## Coding guidelines

1. Please follow [PEP8 guidelines](https://www.python.org/dev/peps/pep-0008/).
2. Create a python virtual environment (virtualenv) for development.
3. Make sure that pylint passes. You may disable specific warnings within the code where
   it is reasonable to do so.
4. Add unit tests for new and modified code.
5. Make sure all existing and new tests pass.
6. Make sure all docstrings have been added.
7. Make sure all dependencies have been added to requirements.
8. Make sure your code works for all required versions of Python.
9. Make sure your code works for all required operating systems.
10. CI is enabled: for documentation changes/linting commits you may include [ci skip]
    in your commit messages. A reminder that CI checks are required before merge!

## Unit Tests

In DeepReg, we use unit tests to ensure a certain code quality and to facilitate the
code maintenance. Following are several guidelines of test writing to ensure a
consistency of code style.

### Test style

we use [pytest](https://docs.pytest.org/en/stable/) for unit tests, please do not use
the package [unittest](https://docs.python.org/3/library/unittest.html).

As we are comparing often the numpy arrays and tensorflow tensors, two functions
`is_equal_np` and `is_equal_tf` are provided in
[test/unit/util.py](https://github.com/DeepRegNet/DeepReg/blob/master/test/unit/util.py).
They will first convert inputs to float32 and compare the max of absolute difference
with a threshold at 1e-6. They can be imported using
`from test.unit.util import is_equal_np` so that we do not need one copy per test file.

### Coverage requirement

For a non-tensorflow-involved function, we need to test

> Check `test_load_nifti_file()` in
> [test_nifti_loader.py](https://github.com/DeepRegNet/DeepReg/blob/master/test/unit/test_nifti_loader.py#L12)
> as an example.

- the correctness of inputs and the error handling for unexpected inputs.
- the correctness of outputs given certain inputs.
- the trigger of all errors (`ValueError`, `AssertionError`, etc.).
- the trigger of warnings.

For a tensorflow-involved function, we need to test

> Check `test_resample()` in
> [test_layer_util.py](https://github.com/DeepRegNet/DeepReg/blob/master/test/unit/test_layer_util.py#L107)
> as an example.

- the correctness of inputs and the error handling for unexpected inputs. The minimum
  requirement is to check the shape of input tensors.
- the correctness of outputs given certain inputs if the function involves mathematical
  operations. Otherwise, at least the output tensor shapes have to be correct.
- the trigger of all errors (`ValueError`, `AssertionError`, etc.).
- the trigger of warnings.

For a class, we need to test

- all the functions in the class using the above standards

We are using [Codecov](https://codecov.io/gh/DeepRegNet/DeepReg) to monitor the test
coverage. While checking the report in file mode, generally a line highlighted by red
means it is not covered by test. In other words, this line has never been executed
during tests. Please check the
[documentation](https://docs.codecov.io/docs/viewing-source-code) for more details about
their coverage report.

### Test example

In this section, we provide some minimum examples to help the understanding.

Assuming we have the following function to be tested:

```python
import logging

def subtract(x: int) -> int:
    """
    A function subtracts one from a non-negative integer.
    :param x: a non-negative integer
    :return: x - 1
    """
    if not isinstance(x, int):
        raise ValueError(f"input {x} is not int")
    assert x >= 0, f"input {x} is negative"
    if x == 0:
        logging.warning("input is zero")
    return x - 1
```

The test should be as follows:

- Name should be the tested function/class with prefix `test_`, in this example, it is
  `test_subtract`.
- All cases are separated by a comment briefly explaining the test case.
- Test a working case, e.g. input is 0 and 1.
- Test a failing case and the `assert`, e.g. input is -1. We need to catch the error and
  check the error message if existed.
- Test the `ValueError`, e.g. input is 0.0, a float. We need to catch the error and
  check the error message if existed.
- Verify the warning is trigger, e.g. input is 0.

```python
import pytest

def test_subtract(caplog):
    """test subtract by verifying its input and outputs"""
    # x = 0
    got = subtract(x=0)
    expected = -1
    assert got == expected

    # x > 0
    got = subtract(x=1)
    expected = 0
    assert got == expected

    # x < 0
    with pytest.raises(AssertionError) as err_info:
        subtract(x=-1)
    assert "is negative" in str(err_info.value)

    # x is not int
    with pytest.raises(ValueError) as err_info:
        subtract(x=0.0)
    assert "is not int" in str(err_info.value)

    # detect caplog
    caplog.clear() # clear previous log
    subtract(x=0)
    assert "input is zero" in caplog.text

    # incorrect warning test example
    # caplog.clear()  # uncomment this line will fail the test
    subtract(x=1)  # this line generates no warning
    assert "input is zero" in caplog.text

    # incorrect error test example
    with pytest.raises(AssertionError) as err_info:
        # this line will trigger the Assertion Error
        # comment the following line will fail the test
        subtract(x=-1)
        # the following line will never be executed
        subtract(x=0.0)
    assert "is negative" in str(err_info.value)

```

In the example above, we are testing warning messages with
[pytest caplog fixture](https://docs.pytest.org/en/stable/logging.html). All the
messages are captured in `caplog` which is the input argument of the test function. Be
careful that it is important to clear the caplog using `caplog.clear`. Otherwise, as the
log is accumulated, we might have unexpected performance.

For instance, with the example above:

```python
    # incorrect warning test example
    # caplog.clear()  # uncomment this line will fail the test
    subtract(x=1)  # this line generates no warning
    assert "input is zero" in caplog.text
```

The test will pass but the assertion works only because we have generated. If the
`caplog.clear()` is uncommented, the test will fail.

Moreover, when testing errors, the `assert` is outside of the
`with pytest.raises(ValueError) as err_info:` and we should not put multiple tests
inside the same `with` as only the first error will be captured.

For instance, in the following test example, the second subtract will never be executed
regardless of whether it is correct or not. If this trigger an error, it will never be
captured.

```python
    # incorrect error test example
    with pytest.raises(AssertionError) as err_info:
        # this line will trigger the Assertion Error
        # comment the following line will fail the test
        subtract(x=-1)
        # the following line will never be executed
        subtract(x=0.0)
    assert "is negative" in str(err_info.value)
```

The test will pass because the first subtract raises the desired assertion error. The
test will fail if we comment out the first subtract.

## Pre-commit setup

[pre-commit](https://pre-commit.com/) is recommended for code style consistency before
each commit. Specifically,

- [seed-isort-config](https://github.com/asottile/seed-isort-config) and
  [isort](https://github.com/timothycrosley/isort) to format package imports in python
  files.
- [Black](https://github.com/psf/black) to format python files.
- [Flake8](https://gitlab.com/pycqa/flake8) to perform linting check,
- [Prettier](https://prettier.io/) to format markdown files.

Pre-commit is installed during the package installation via `pip install -e .`. To
activate pre-commit, make sure the git is installed (`sudo apt install git` for linux)
and run `pre-commit install` under the root of this repository `DeepReg/`.

Optionally, use `git commit --no-verify -m "This is a commit message placeholder."` to
skip pre-commit, and use `pre-commit run --all-files` to format files before pull
request.

Sometimes, black might have conflicts with prettier or flake8 and below are some
possible cases and work around

- If a code is followed by a long comment in the same line, Black attempts to break
  lines. So we should put comment in the line above instead.
- For lists/tuples, do not add comma after the last element, unless it's a single
  element tuple, like `(1,)`.

To check if Black is causing conflicts, run `black .` in the root of DeepReg you will
see the formatted files by Black, run `pre-commit run --all-files`, you will see the
final versions. Compare them to understand an issue. If there's a new case and there's
an workaround, please update this section. Otherwise, please raise an issue.

## Documentation Pages

Below are instructions for creating and modifying pages on this website, as found within
the `docs` folder of the repository.

### Linux

- [Docsify](https://docsify.js.org/) converts markdown files into pages and they are
  hosted on github page.
- Use `cd docs && python -m http.server` to visualize the pages locally.
- The required package
  [simple_http_server](https://github.com/keijack/python-simple-http-server) has been
  installed during the package installation via `pip install -e .`.

### Windows

- Install [Jekyll](https://jekyllrb.com/docs/installation/windows/) via RubyInstaller,
  and follow the install instructions on that page.
- In a Windows Command Prompt (WCP) started with Ruby, use `gem install github-pages` to
  install the required packages.
- Next, the same WCP - navigate to the `docs` folder in `DeepReg`, and use
  `jekyll serve` to create a directory located at `../docs/_site`.
  - You should now see a local version of the site at http://127.0.0.1:4000.
  - You will need to work in the newly created `../docs/_site` folder to make your
    changes, and copy them back to `../docs` in order to commit them here.

_Note:_ Make sure to frequently update the GitHub pages gem with
`gem update github-pages`. There are dependencies and various packages that are updated
frequently - and the website may not display locally the same as it will when pushed if
you do not have the most recent versions.

### MacOS

TBC

## Contributing to a DeepReg Demo

Please see details in the [demo requirement](doc_demo_requirement.md).
