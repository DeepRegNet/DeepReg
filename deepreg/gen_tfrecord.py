"""
Module to implement command line interface and run tf
record function
"""
import argparse
import os
import shutil

import deepreg.config.parser as config_parser
import deepreg.dataset.load as load
from deepreg.dataset.tfrecord import write_tfrecords


def main(args=None):
    """Entry point for gen_tfrecord"""

    parser = argparse.ArgumentParser(description="gen_tfrecord")

    ## ADD POSITIONAL ARGUMENTS
    parser.add_argument(
        "--config_path", "-c", help="Path of config", type=str, required=True
    )

    parser.add_argument(
        "--examples_per_tfrecord",
        "-n",
        help="Number of examples per tfrecord",
        type=int,
        default=64,
    )

    args = parser.parse_args(args)

    config = config_parser.load(args.config_path)
    data_config = config["data"]
    tfrecord_dir = data_config["tfrecord_dir"]
    data_config["tfrecord_dir"] = ""

    if os.path.exists(tfrecord_dir) and os.path.isdir(tfrecord_dir):
        remove = input("%s exists. Remove it or not? Y/N\n" % tfrecord_dir)
        if remove.lower() == "y":
            shutil.rmtree(tfrecord_dir)
    for mode in ["train", "valid", "test"]:
        data_loader = load.get_data_loader(data_config, mode)
        write_tfrecords(
            data_dir=os.path.join(tfrecord_dir, mode),
            data_generator=data_loader.data_generator(),
            examples_per_tfrecord=args.examples_per_tfrecord,
        )


if __name__ == "__main__":
    main()
