# Commit

To facilitate review of contributions, we encourage contributors to adhere to the
following commit guidelines.

1. The commit message should start with `Issue #<issue number>:` so that the commits are
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
