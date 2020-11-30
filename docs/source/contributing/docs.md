# Documentation

We use [Sphinx](https://www.sphinx-doc.org/en/master/) to organize the documentation and
it is hosted in [ReadTheDocs](https://readthedocs.org/projects/deepreg/).

## Local Build

Please run the following command under `docs/` directory for generating the
documentation pages locally. Generated files are under `docs/build/html/`

```bash
make clean html
```

where

- `clean` removes the possible built files.
- Optionally we can add `SPHINXOPTS="-W"` to fail on any warnings, but we are currently
  having `document isn't included in any toctree` warning and no better solution has
  been found yet.

## Recommendations

There are some recommendations regarding the docs.

- **We prefer markdown files** over reStructuredText files as its linting is covered
  using [Prettier](https://prettier.io/).

  Only use reStructuredText (rst) files for some functionalities not supported by
  markdown, such as

  - `toctree`
  - warning/notes boxes in [Installation](../getting_started/install.html)

  The conversion between markdown and rst can be done automatically using free online
  tool [Pandoc](https://pandoc.org/try/).

- When linking to other pages, **please use relative paths** such as
  `../getting_started/install.html` instead of absolute paths
  `https://deepreg.readthedocs.io/en/latest/getting_started/install.html` as relative
  paths are more robust for different version of documentations.

- To refer a markdown file outside of the source folder, create an rst file and use
  `.. mdinclude:: <makrdown file path>` to include the markdown source.

  Check the source code of
  [paired lung CT image registration page](../demo/paired_ct_lung.html) as an example.
