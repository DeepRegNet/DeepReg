# Testing

In DeepReg, we use [pytest](https://docs.pytest.org/en/stable/) (not
[unittest](https://docs.python.org/3/library/unittest.html)) for unit tests to ensure a
certain code quality and to facilitate the code maintenance.

The testing is checked via [Travis-CI](https://travis-ci.org/github/DeepRegNet/DeepReg)
and [Codecov](https://codecov.io/gh/DeepRegNet/DeepReg) is used to monitor the test
coverage. While checking the Codecov report in file mode, generally a line highlighted
by red means it is not covered by test. In other words, this line has never been
executed during tests. Please check the
[Codecov documentation](https://docs.codecov.io/docs/viewing-source-code) for more
details about their coverage report.

Following are the guidelines of test writing to ensure a consistency of code style.

## Coverage requirement

### Class

For a class, we need to test all the functions in the class.

### Non-TensorFlow function

For a non-TensorFlow function, we need to test

- The correctness of inputs and the error handling for unexpected inputs.
- The correctness of outputs given certain inputs.
- The trigger of all errors (`ValueError`, `AssertionError`, etc.).
- The trigger of warnings.

Check `test_load_nifti_file()` in
[test_nifti_loader.py](https://github.com/DeepRegNet/DeepReg/blob/main/test/unit/test_nifti_loader.py#L12)
as an example.

### TensorFlow function

For a TensorFlow-involved function, we need to test

- The correctness of inputs and the error handling for unexpected inputs. The minimum
  requirement is to check the shape of input tensors.
- The correctness of outputs given certain inputs if the function involves mathematical
  operations. Otherwise, at least the output tensor shapes have to be correct.
- The trigger of all errors (`ValueError`, `AssertionError`, etc.).
- The trigger of warnings.

Check `test_resample()` in
[test_layer_util.py](https://github.com/DeepRegNet/DeepReg/blob/main/test/unit/test_layer_util.py#L107)
as an example.

### Helper functions

As we are comparing often the numpy arrays and TensorFlow tensors, two functions
`is_equal_np` and `is_equal_tf` are provided in
[test/unit/util.py](https://github.com/DeepRegNet/DeepReg/blob/main/test/unit/util.py).
They will first convert inputs to float32 and compare the max of absolute difference
with a threshold at 1e-6. They can be imported using
`from test.unit.util import is_equal_np` so that we do not need one copy per test file.

## Example unit test

In this section, we provide some minimum examples to help the understanding.

### Function to be tested

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

### Coverage requirement

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

### Test code

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

### Common errors

**Forget to clear caplog**

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

**Test multiple errors together**

When testing errors, the `assert` should be outside of the
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
