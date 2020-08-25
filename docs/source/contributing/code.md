# Coding

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
