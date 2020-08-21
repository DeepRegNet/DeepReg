# Issues and Pull requests

We welcome contributions to DeepReg.

## Reporting bugs and feature requests

Found a bug? Create a [new issue](https://github.com/DeepRegNet/DeepReg/issues/new)!

When reporting a bug, please include:

1. The version of DeepReg you are using
2. Your OS version (for example Windows 10 64-bit, macOS High Sierra, Ubuntu 18.04)
3. Detailed steps to reproduce the bug.

## Fixing bugs or implement features

The easiest way to contribute is to follow these guidelines:

1. Look through the [issues](https://github.com/DeepRegNet/DeepReg/issues) and assign
   the relevant issue to yourself. If there is not an existing issue that covers your
   work, please create [one](https://github.com/DeepRegNet/DeepReg/issues/new).
2. Read the design considerations below.
3. [Fork the repository](https://github.com/DeepRegNet/DeepReg/forks/new)
4. Create a branch for your changes. The branch name should start with the issue number,
   followed by hyphen separated words describing the issue, e.g.
   `1-update-contribution-guidelines`.
5. Make your changes following the [coding guidelines](code.md).
6. Commit and push your changes to your fork.

   The commit message should start with `Issue #<issue number>`, for example: "Issue #1:
   Fixed typo".

   Commit in small, related chunks. Review each commit and explain its purpose in the
   commit message. Refer to the commit style section below for a more detailed guide.

7. [Submit a pull request](https://github.com/DeepRegNet/DeepReg/merge-requests/new)
8. Pull request will be reviewed and, if necessary, changes will be suggested.

## Commit style

To facilitate review of contributions, we encourage contributors to adhere to the
following commit guidelines.

1. The commit message should start with `Issue #<issue number>` so that the commits are
   tied to a specific issue.

   - **Good**:

     ```text
     Issue #1: modified resample function docstring to reflect changes in function args
     ```

   - **Moderate/OK**:

     Inconsistent commit style.

     ```text
     ref #<issue number>: modified resample function docstring to reflect changes in function args
     ```

   - **Bad**:

     Missing issue number.

     ```text
     modified resample function docstring to reflect changes in function args
     ```

2) Include related changes in the same commit, where possible, For example, if multiple
   typos are spotted in a document, aim to resolve them in the same commit instead of
   separate commits. This improves history readability.

   - **Good**:

     One commit for the same type of problem

     ```text
     Issue #<issue number>: fixed typos in function x
     ```

   - **Bad**:

     Multiple commits of the same message in the same thread. Clutters repo history.

3) Strive to add informative commit messages.

   - **Good**:

     ```text
     Issue #<issue number>: removed unused arguments across function x in loops to comply with PEP8 standard in file y
     Issue #<issue number>: added new data loader class inheriting from base class to deal with np array file format
     ```

   - **Not acceptable**:

     Not enough details, hard to tell explicitly what was changed without doing an in
     depth review.

     ```text
     Issue #<issue number>: lint
     Issue #<issue number>: add loader
     ```
