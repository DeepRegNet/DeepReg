# Visualisation tool

DeepReg provides a visuaisation tool which allows the user to generate various
visualisations from nifti images. The tool is compatible with outputs from
deepreg_predict as well as with other nifti images.

The visualisation tool currently offers four functionalities using the command
`deepreg_vis`, with their Python functions explained in the final section of this
document. Run the examples below after changing directory to the root folder `DeepReg`.

The creation of `.gif` files requires a movie writer that is compatible with
`matplotlib` like `ffmpeg` in order to be able to write `.gif` files. Please refer to
the [matplotlib documentation](https://matplotlib.org/3.3.1/api/animation_api.html) for
more details about writers. The `ffmpeg` writer was used to test functionality of this
visualisation tool, however, other writers can also be installed and used.

## General arguments for `deepreg_vis`

- `-m` or `--mode`: This specifies which mode to use to generate the visualisation. See
  below for available modes.
- `-i` or `--image-paths`: This is the path of the image or images that need to be used
  to generate the visualisation. Multiple paths can be passed using a comma separated
  string.
- `-s` or `--save-path`: This is the path to the directory where the visualisation will
  be saved. The name of the visualisation is auto generated, however, if `--fname`
  argument is available for a specific mode, then a custom filename can be specified.
  (default results in saving to current directory)

## GIF over image slices

This creates an animation which is an iteration through the image slices of a 3D image,
can be accessed by passing `--mode 0` or `-m 0`.

In addition to the general arguments, the additional arguments applicable to this mode
are:

- `--interval`: This argument is optional and can be used to specify the time, in
  milliseconds, between successive frames of an animation. (default=50)

The output will be a file with the same name as the original with a `.gif` extension. If
multiple image paths are passed in the `-i` or `--image-paths` argument then multiple
`.gif` files will be generated, one for each unique image path passed.

A simple example, which takes an image path and saves a `.gif` animation, is shown
below:

```bash
deepreg_vis -m 0 -i ./data/test/nifti/unit_test/moving_image.nii.gz -s logs
```

## GIF that shows warping

This functionality produces an animation showing warping for a single image slice using
a ddf, can be accessed by passing `--mode 1` or `-m 1`.

In addition to the general arguments, the additional arguments applicable to this mode
are:

- `--ddf-path`: This argument is required for this mode and specifies the path of the
  ddf to use for warping the image.
- `--slice-inds`: This argument is optional and can be used to specify the indexes to be
  used to generate the visualisation. Multiple indexes can be passed by using a comma
  separated string. (default results in a random slice being used)
- `--interval`: This argument is optional and can be used to specify the time, in
  milliseconds, between successive frames of an animation. (default=50)
- `--num-interval`: The number of intervals to use for warping. (default=100)

The output will be a file with the slice number appended to the original file name, with
a `.gif` extension. For example if a file is named `moving_image.nii.gz` and slice
number 2 and 3 three are chosen, the files produced will be `moving_image_slice_2.gif`
and `moving_image_slice_3.gif`.If multiple image paths are passed in the `-i` or
`--image-paths` argument then multiple `.gif` files will be generated, if multiple slice
indexes are specified then multiple `.gif` files are generated for each unique image
path passed.

A simple example, which takes an image, slice indexes and a ddf path and saves a `.gif`
animation for each slice, is shown below:

```bash
deepreg_vis -m 1 -i ./data/test/nifti/unit_test/moving_image.nii.gz --ddf-path "./data/test/nifti/unit_test/ddf.nii.gz" --slice-inds '2,3' -s logs
```

## Plot of image slices

This functionality produces a plot of image slices from a single or multiple images.
Each column is a different image and each row is a different slice, can be accessed by
passing `--mode 2` or `-m 2`.

In addition to the general arguments, the additional arguments applicable to this mode
are:

- `--slice-inds`: This argument is optional and can be used to specify the indexes to be
  used to generate the visualisation. Multiple can be passed by using a comma separated
  string. (default results in a random slice being used)
- `--col-titles`: This is optional. The title of the column to be used in order from
  left to right column. (default results in using file name specified in image path as
  column name)
- `--fname`: Optional argument of file name to save visualisation to; should end with an
  appropriate file extension like `.png` or `.jpeg`. (default='visualisation.png')

The output is a single file which contains a static visualisation. The visualisation is
different images in the columns and different slices in the rows.

A simple example, which takes three images and three slice indexes and saves a `.png`
file, is shown below (this will create a plot with 3 columns and 3 rows):

```
deepreg_vis -m 2 -i './data/test/nifti/unit_test/moving_image.nii.gz, ./data/test/nifti/unit_test/moving_image.nii.gz, ./data/test/nifti/unit_test/moving_image.nii.gz' --slice-inds '2,3,4' -s logs
```

## Tiled GIF over image slices

This functionality produces an animation with multiple animated images that are tiled
together, can be accessed by passing `--mode 3` or `-m 3`. The images used as input must
all be of the same size.

- `--interval`: This argument is optional and can be used to specify the time, in
  milliseconds, between successive frames of an animation. (default=50)
- `--size`: This is an optional argument and can be used to specify the number of rows
  and columns for the final tiled animation. For example `'2, 3'` means 2 rows and 3
  columns, in this case 6 images must be passed for the visualisation. (default='2,2')
- `--fname`: Optional argument of file name to save visualisation to; should end with
  `.gif`. (default='visualisation.gif')

The output is a single file which contains the tiled animation. The image paths must be
passed in the order in which they are to be tiled (order is from left to right and then
next row).

A simple example, which takes four images and creates an animation over the slices of
the four images in a tiled manner, is shown below:

```bash
deepreg_vis -m 3 -i './data/test/nifti/unit_test/moving_image.nii.gz, ./data/test/nifti/unit_test/moving_image.nii.gz, ./data/test/nifti/unit_test/moving_image.nii.gz, ./data/test/nifti/unit_test/moving_image.nii.gz' --size '2,2' -s logs
```

## Using the same functionality by importing python functions

The same functionality can be accessed via python functions.

The names of the functions are:

- `gif_slices`: equivalent to `--mode 0` or `-m 0`
- `gif_warp`: equivalent to `--mode 1` or `-m 1`
- `tile_slices`: equivalent to `--mode 2` or `-m 2`
- `gif_tile_slices`: equivalent to `--mode 3` or `-m 3`

The functions can be imported in a python script as follows:

```
from deepreg.vis import function_name
```

For usage details please see the function docstrings. Function docstrings can be
accessed in a python prompt by:

```
help(func_name)
```
