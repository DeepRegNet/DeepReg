# coding=utf-8

"""
Module to warp a image with given ddf. A CLI tool is provided.
"""

import argparse
import logging
import os

import nibabel as nib
import numpy as np
import tensorflow as tf

from deepreg.dataset.loader.nifti_loader import load_nifti_file
from deepreg.model.layer_util import warp_image_ddf


def warp(image_path: str, ddf_path: str, out_path: str):
    """
    :param image_path: file path of the image file
    :param ddf_path: file path of the ddf file
    :param out_path: file path of the output
    """
    if out_path == "":
        out_path = "warped.nii.gz"
        logging.warning(
            f"Output file path is not provided, will save output in {out_path}."
        )
    else:
        if not (out_path.endswith(".nii") or out_path.endswith(".nii.gz")):
            out_path = os.path.join(os.path.dirname(out_path), "warped.nii.gz")
            logging.warning(
                f"Output file path should end with .nii or .nii.gz, "
                f"will save output in {out_path}."
            )
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # load image and ddf
    image = load_nifti_file(image_path)
    ddf = load_nifti_file(ddf_path)
    if len(image.shape) not in [3, 4]:
        raise ValueError(
            f"image shape must be (m_dim1, m_dim2, m_dim3) "
            f"or (m_dim1, m_dim2, m_dim3, ch),"
            f" got {image.shape}"
        )
    if not (len(ddf.shape) == 4 and ddf.shape[-1] == 3):
        raise ValueError(
            f"ddf shape must be (f_dim1, f_dim2, f_dim3, 3), got {ddf.shape}"
        )
    # add batch dimension manually
    image = tf.expand_dims(image, axis=0)
    ddf = tf.expand_dims(ddf, axis=0)

    # warp
    warped_image = warp_image_ddf(image=image, ddf=ddf, grid_ref=None)
    warped_image = warped_image.numpy()
    warped_image = warped_image[0, ...]  # removed added batch dimension

    # save output
    nib.save(img=nib.Nifti2Image(warped_image, affine=np.eye(4)), filename=out_path)


def main(args=None):
    """Entry point for warp script."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image", "-i", help="File path for image file", type=str, required=True
    )

    parser.add_argument(
        "--ddf", "-d", help="File path for ddf file", type=str, required=True
    )

    parser.add_argument("--out", "-o", help="Output path for warped image", default="")

    # init arguments
    args = parser.parse_args(args)
    warp(image_path=args.image, ddf_path=args.ddf, out_path=args.out)


if __name__ == "__main__":
    main()
