# Release

DeepReg is distributed on PyPI. To create new releases, you can follow the below
instructions to automatically submit new versions to PyPI via our GitHub Actions
workflow.

## Creating a new Release

From the main DeepReg repository page, head to
[releases](https://github.com/DeepRegNet/DeepReg/releases). From here, you can
[draft a new release](https://github.com/DeepRegNet/DeepReg/releases/new).

## Tagging & Titling

We follow [semver](https://semver.org/) naming conventions for tags, with `vX.Y.Z` where
each represents major, minor, and patch release versions.

From semver.org:

> Major version: when you make incompatible API changes,
>
> Minor version: when you add functionality in a backwards compatible manner, and
>
> Patch version: when you make backwards compatible bug fixes.

Typically, most releases will be an increment of the minor or patch versions.

Enter the new version in the format `vX.Y.Z` into the "Tag version" and "Release title"
fields of the draft a release page.

## Publish!

Click the "Publish release" button, and our GitHub Action workflow will handle the rest.

It's recommended that you check the
[output log](https://github.com/DeepRegNet/DeepReg/actions) from the GitHub Actions page
to make sure everything went as planned for the release.
