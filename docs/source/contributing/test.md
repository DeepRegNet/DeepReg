# Unit Test

In DeepReg, we use [pytest](https://docs.pytest.org/en/stable/) (not
[unittest](https://docs.python.org/3/library/unittest.html)) for unit tests to ensure a
certain code quality and to facilitate the code maintenance.

For testing the code locally,

- If you only need to test with python 3.7, please execute `pytest` at the repository
  root:

  ```bash
  pytest test/
  ```

- If you want to test with all supported python versions, please execute `tox` at the
  repository root:

  ```bash
  tox
  ```

Moreover, the testing is checked automatically via
[GitHub workflows](https://github.com/DeepRegNet/DeepReg/actions) and
[Codecov](https://codecov.io/gh/DeepRegNet/DeepReg) is used to monitor the test
coverage. While checking the Codecov report in file mode, generally a line highlighted
by red means it is not covered by test. Please check the
[Codecov documentation](https://docs.codecov.io/docs/viewing-source-code) for more
details.

## Test requirement

We would like to achieve 100% test coverage. In general, tests should be

- thorough, covering different scenarios.
- independent, different scenarios are not tested together.
- clean and compact, for instance,
  - Use [parameterized test](https://docs.pytest.org/en/stable/example/parametrize.html)
    to reduce code redundancy.
  - Use `is_equal_np` and `is_equal_tf` provided in
    [test/unit/util.py](https://github.com/DeepRegNet/DeepReg/blob/main/test/unit/util.py)
    to compare arrays or tensors.

The detailed requirements are as follows:

- Test all functions in python classes.
- Test the trigger of all warning and errors in functions.
- Test the correctness of output values for functions.
- Test at least the correctness of output shapes for TensorFlow functions.

## Example unit test

We provide here an example to help understanding the requirements.

```python
import pytest


def subtract(x: int) -> int:
    """
    A function subtracts one from a non-negative integer.
    :param x: a non-negative integer
    :return: x - 1
    """
    assert isinstance(x, int), f"input {x} is not int"
    assert x >= 0, f"input {x} is negative"
    return x - 1


class TestSubtract:
    @pytest.mark.parametrize("x,expected",[(0, -1), (1,0)])
    def test_value(self, x, expected):
        got = subtract(x=x)
        assert got == expected

    @pytest.mark.parametrize("x,msg", [(-1, "is negative"), (0.0, "is not int")])
    def test_err(self, x, msg):
        with pytest.raises(AssertionError) as err_info:
            subtract(x=x)
        assert msg in str(err_info.value)
```

where

- we group multiple test functions for `subtract` under the same class `TestSubtract`.
- we [parameterize test](https://docs.pytest.org/en/stable/example/parametrize.html) to
  test different inputs.
- we catch errors using `pytest.raises` and check error messages.

For further usage like [fixture](https://docs.pytest.org/en/stable/fixture.html) and
other functionalities, please check
[pytest documentation](https://docs.pytest.org/en/stable/index.html) or
[existing tests](https://github.com/DeepRegNet/DeepReg/tree/main/test/unit) in DeepReg.
You can also [raise an issue](https://github.com/DeepRegNet/DeepReg/issues/new/choose)
for any questions.
