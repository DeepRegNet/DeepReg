# Packaging a Release

`DeepReg` is disributed on PyPI. To create new releases, you can follow the below
instructions and submit new versions to PyPI.

## Prerequisites

Make sure you have `setuptools`, `wheel`, and `twine` installed in your environment:

```bash
pip install setuptools wheel twine
```

Update your `setup.py` file with the appropriate version number. Then, from within the
`DeepReg` folder where you want the version to be built from:

```bash
python setup.py sdist bdist_wheel
```

## Upload to TestPyPI

Once built, upload to `testpypi` to ensure that the package runs as expected prior to
adding the new release to `pypi`. This will require an account on `testpypi`.

```bash
twine upload --repository testpypi dist/*
```

You'll be prompted to enter your username and password.

## Upload to PyPI

If all works well, upload to `pypi`:

```bash
twine upload dist/*
```

You'll, once again, be prompted to enter your username and password.

## Tag & upload the release to GitHub

Make sure to
[add the release to the DeepReg repository on GitHub](https://github.com/DeepRegNet/DeepReg/releases)
as well. Make sure to follow the given naming conventions for tags with `vX.Y.Z` with
major, minor, and batch releases.
