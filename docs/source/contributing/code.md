# Coding

To ensure the code quality and the consistency, we recommend the following guidelines.

## Coding design

1. Please use packages that already have been included. Additional libraries will
   require a longer review for scrutinizing its necessity.

   In case of adding a new dependency, please make sure all dependencies have been added
   to `requirements.txt`.

1. Please prevent adding redundant code. Try wrapping the code block into a function or
   class instead.

   In case of adding new functions, please make sure the corresponding unit tests are
   added.

1. When adding/modifying functions/classes, please make sure the corresponding
   docstrings are added/updated.

1. Please check [Unit test requirement](test.html) for detailed requirements on unit
   testing.

1. Please check [Documentation requirement](docs.html) for detailed requirements on
   documentation,
