#!/usr/bin/python3
import argparse
import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile


def download(dirs, output_dir="./"):
    """
    Downloads the files and directories from DeepReg into `output_dir`, keeping only `dirs`.
    """

    output_dir = os.path.abspath(output_dir)  # Get the output directory.

    print("\nWill download folders: [", *dirs, sep=" ", end=" ")
    print("] into {}.\n".format(output_dir))

    print("Downloading archive from DeepReg repository...\n")

    response = urlopen(
        "https://github.com/DeepRegNet/DeepReg/archive/main.zip"
    )  # Download the zip.

    print("Downloaded archive. Extracting files...\n")

    with ZipFile(BytesIO(response.read())) as zf:

        pathnames = zf.namelist()
        head = pathnames[0]
        keepdirs = [
            os.path.join(head, d) for d in dirs
        ]  # Find our folders to keep, based on what user specifies.

        for pathname in pathnames:
            if any(d in pathname for d in keepdirs):

                info = zf.getinfo(pathname)
                info.filename = info.filename.replace(
                    head, ""
                )  # Remove head directory from filepath
                zf.extract(info, output_dir)

                print("Downloaded {}".format(info.filename))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        "-d",
        dest="output_dir",
        default="./",
        help="All directories will be downloaded to the specified directory.",
    )
    args = parser.parse_args()

    dirs = [
        "config",
        "data",
        "demos",
    ]

    download(dirs, args.output_dir)

    print(
        "\nDownload complete. Please refer to the DeepReg Quick Start guide for next steps."
    )


if __name__ == "__main__":
    main()
