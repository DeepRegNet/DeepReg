"""
Module to generate visualisations of data
at command line interface.
Requires ffmpeg writer to write gif files
"""

import argparse
import os
from typing import List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib

from deepreg import log
from deepreg.dataset.loader.nifti_loader import load_nifti_file
from deepreg.model.layer import Warping

logger = log.get(__name__)


def string_to_list(string: str) -> List[str]:
    """
    Converts a comma separated string to a list of strings
    also removes leading or trailing spaces from each element in list.

    :param string: string which is to be converted to list
    :return: list of strings
    """
    return [elem.strip() for elem in string.split(",")]


def gif_slices(img_paths, save_path="", interval=50):
    """
    Generates and saves gif of slices of image
    supports multiple images to generate multiple gif files.

    :param img_paths: list or comma separated string of image paths
    :param save_path: path to directory where visualisation/s is/are to be saved
    :param interval: time in miliseconds between frames of gif
    """
    if type(img_paths) is str:
        img_paths = string_to_list(img_paths)
    img = load_nifti_file(img_paths[0])
    img_shape = np.shape(img)
    for img_path in img_paths:
        fig = plt.figure()
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        fig.add_axes(ax)

        img = load_nifti_file(img_path)

        frames = []
        for index in range(img_shape[-1]):
            frame = plt.imshow(img[:, :, index], aspect="auto", animated=True)
            # plt.axis('off')
            frames.append([frame])

        anim = animation.ArtistAnimation(fig, frames, interval=interval)

        path_to_anim_save = os.path.join(
            save_path, os.path.split(img_path)[-1].split(".")[0] + ".gif"
        )

        anim.save(path_to_anim_save)
        logger.info("Animation saved to: %s.", path_to_anim_save)


def tile_slices(img_paths, save_path="", fname=None, slice_inds=None, col_titles=None):
    """
    Generate a tiled plot of multiple images for multiple slices in the image.
    Rows are different slices, columns are different images.

    :param img_paths: list or comma separated string of image paths
    :param save_path: path to directory where visualisation/s is/are to be saved
    :param fname: file name with extension to save visualisation to
    :param slice_inds: list of slice indices to plot for each image
    :param col_titles: titles for each column, if None then inferred from file names
    """
    if type(img_paths) is str:
        img_paths = string_to_list(img_paths)
    img = load_nifti_file(img_paths[0])
    img_shape = np.shape(img)

    if slice_inds is None:
        slice_inds = [round(np.random.rand() * (img_shape[-1]) - 1)]

    if col_titles is None:
        col_titles = [
            os.path.split(img_path)[-1].split(".")[0] for img_path in img_paths
        ]

    num_inds = len(slice_inds)
    num_imgs = len(img_paths)

    subplot_mat = np.array(np.arange(num_inds * num_imgs) + 1).reshape(
        num_inds, num_imgs
    )

    plt.figure(figsize=(num_imgs * 2, num_inds * 2))

    imgs = [load_nifti_file(p) for p in img_paths]

    for col_num, img in enumerate(imgs):
        for row_num, index in enumerate(slice_inds):
            plt.subplot(num_inds, num_imgs, subplot_mat[row_num, col_num])
            plt.imshow(img[:, :, index])
            plt.axis("off")
            if row_num - 0 < 1e-3:
                plt.title(col_titles[col_num])

    if fname is None:
        fname = "visualisation.png"
    save_fig_to = os.path.join(save_path, fname)
    plt.savefig(save_fig_to)
    logger.info("Plot saved to: %s", save_fig_to)


def gif_warp(
    img_paths, ddf_path, slice_inds=None, num_interval=100, interval=50, save_path=""
):
    """
    Apply ddf to image slice/s to generate gif.

    :param img_paths: list or comma separated string of image paths
    :param ddf_path: path to ddf to use for warping
    :param slice_inds: list of slice indices to use for each image
    :param num_interval: number of intervals in which to apply ddf
    :param interval: time in miliseconds between frames of gif
    :param save_path: path to directory where visualisation/s is/are to be saved
    """
    if type(img_paths) is str:
        img_paths = string_to_list(img_paths)

    image = load_nifti_file(img_paths[0])
    img_shape = np.shape(image)

    if slice_inds is None:
        slice_inds = [round(np.random.rand() * (img_shape[-1]) - 1)]

    for img_path in img_paths:
        for slice_ind in slice_inds:

            fig = plt.figure()
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)

            ddf_scalers = np.linspace(0, 1, num=num_interval)

            frames = []
            for ddf_scaler in ddf_scalers:
                image = load_nifti_file(img_path)
                ddf = load_nifti_file(ddf_path)
                fixed_image_shape = ddf.shape[:3]
                image = np.expand_dims(image, axis=0)
                ddf = np.expand_dims(ddf, axis=0) * ddf_scaler

                warped_image = Warping(fixed_image_size=fixed_image_shape)([ddf, image])
                warped_image = np.squeeze(warped_image.numpy())

                frame = plt.imshow(
                    warped_image[:, :, slice_ind], aspect="auto", animated=True
                )

                frames.append([frame])

            anim = animation.ArtistAnimation(fig, frames, interval=interval)
            path_to_anim_save = os.path.join(
                save_path,
                os.path.split(img_path)[-1].split(".")[0]
                + "_slice_"
                + str(slice_ind)
                + ".gif",
            )

            anim.save(path_to_anim_save)
            logger.info("Animation saved to: %s", path_to_anim_save)


def gif_tile_slices(img_paths, save_path=None, size=(2, 2), fname=None, interval=50):
    """
    Creates tiled gif over slices of multiple images.

    :param img_paths: list or comma separated string of image paths
    :param save_path: path to directory where visualisation/s is/are to be saved
    :param interval: time in miliseconds between frames of gif
    :param size: number of columns and rows of images for the tiled gif
        (tuple e.g. (2,2))
    :param fname: filename to save visualisation to
    """
    if type(img_paths) is str:
        img_paths = string_to_list(img_paths)

    num_images = np.prod(size)
    if int(len(img_paths)) != int(num_images):
        raise ValueError(
            "The number of images supplied is "
            + str(len(img_paths))
            + " whereas the number required is "
            + str(np.prod(size))
            + " as size specified is "
            + str(size)
        )

    img = load_nifti_file(img_paths[0])
    img_shape = np.shape(img)

    imgs = []
    for img_path in img_paths:
        img = load_nifti_file(img_path)
        shape = np.shape(img)
        if shape != img_shape:
            raise ValueError("all images do not have equal shapes")
        imgs.append(img)

    frames = []

    fig = plt.figure()
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)

    for index in range(img_shape[-1]):

        temp_tiles = []
        frame = np.matlib.repmat(
            np.ones((img_shape[0], img_shape[1])), size[0], size[1]
        )

        for img in imgs:
            temp_tile = img[:, :, index]
            temp_tiles.append(temp_tile)

        tile_count = 0
        for i in range(size[0]):
            for j in range(size[1]):
                tile = temp_tiles[tile_count]
                tile_count += 1
                frame[
                    i * img_shape[0] : (i + 1) * img_shape[0],
                    j * img_shape[0] : (j + 1) * img_shape[0],
                ] = tile

        frame = plt.imshow(frame, aspect="auto", animated=True)

        frames.append([frame])

    if fname is None:
        fname = "visualisation.gif"

    anim = animation.ArtistAnimation(fig, frames, interval=interval)
    path_to_anim_save = os.path.join(save_path, fname)

    anim.save(path_to_anim_save)
    logger.info("Animation saved to: %s", path_to_anim_save)


def main(args=None):
    """
    CLI for deepreg_vis tool.

    Requires ffmpeg wirter to write gif files.

    :param args:
    """
    parser = argparse.ArgumentParser(
        description="deepreg_vis", formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--mode",
        "-m",
        help="Mode of visualisation \n"
        "0 for animtion over image slices, \n"
        "1 for warp animation, \n"
        "2 for tile plot",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--image-paths",
        "-i",
        help="File path for image file "
        "(can specify multiple paths using a comma separated string)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--save-path",
        "-s",
        help="Path to directory where resulting visualisation is saved",
        default="",
    )

    parser.add_argument(
        "--interval",
        help="Interval between frames of animation (in miliseconds)\n"
        "Applicable only if --mode 0 or --mode 1 or --mode 3",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--ddf-path",
        help="Path to ddf used for warping images\n"
        "Applicable only and required if --mode 1",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--num-interval",
        help="Number of intervals to use for warping\n" "Applicable only if --mode 1",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--slice-inds",
        help="Comma separated string of indexes of slices"
        " to be used for the visualisation\n"
        "Applicable only if --mode 1 or --mode 2",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--fname",
        help="File name (with extension like .png, .jpeg, .gif, ...)"
        " to save visualisation to\n"
        "Applicable only if --mode 2 or --mode 3",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--col-titles",
        help="Comma separated string of column titles to use "
        "(inferred from file names if not provided)\n"
        "Applicable only if --mode 2",
        default=None,
    )
    parser.add_argument(
        "--size",
        help="Comma separated string of number of columns and rows (e.g. '2,2')\n"
        "Applicable only if --mode 3",
        default="2,2",
    )

    # init arguments
    args = parser.parse_args(args)

    if args.slice_inds is not None:
        args.slice_inds = string_to_list(args.slice_inds)
        args.slice_inds = [int(elem) for elem in args.slice_inds]

    if args.mode == 0:
        gif_slices(
            img_paths=args.image_paths, save_path=args.save_path, interval=args.interval
        )
    elif args.mode == 1:
        if args.ddf_path is None:
            raise Exception("--ddf-path is required when using --mode 1")
        gif_warp(
            img_paths=args.image_paths,
            ddf_path=args.ddf_path,
            slice_inds=args.slice_inds,
            num_interval=args.num_interval,
            interval=args.interval,
            save_path=args.save_path,
        )
    elif args.mode == 2:
        tile_slices(
            img_paths=args.image_paths,
            save_path=args.save_path,
            fname=args.fname,
            slice_inds=args.slice_inds,
            col_titles=args.col_titles,
        )
    elif args.mode == 3:
        size = tuple([int(elem) for elem in string_to_list(args.size)])
        gif_tile_slices(
            img_paths=args.image_paths,
            save_path=args.save_path,
            fname=args.fname,
            interval=args.interval,
            size=size,
        )


if __name__ == "__main__":
    main()  # pragma: no cover
