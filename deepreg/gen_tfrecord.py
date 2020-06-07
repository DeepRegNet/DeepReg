import os
import shutil

import click

import deepreg.config.parser as config_parser
import deepreg.data.load as load
from deepreg.data.tfrecord import write_tfrecords


@click.command()
@click.option(
    "--config_path", "-c",
    help="Path of config",
    type=click.Path(file_okay=True, dir_okay=False, exists=True),
    required=True,
)
@click.option(
    "--examples_per_tfrecord", "-n",
    help="Number of examples per tfrecord",
    type=int,
    default=64,
)
def main(config_path, examples_per_tfrecord):
    config = config_parser.load(config_path)
    data_config = config["data"]
    tfrecord_dir = data_config["tfrecord_dir"]
    data_config["tfrecord_dir"] = ""

    if os.path.exists(tfrecord_dir) and os.path.isdir(tfrecord_dir):
        remove = input("%s exists. Remove it or not? Y/N\n" % tfrecord_dir)
        if remove.lower() == "y":
            shutil.rmtree(tfrecord_dir)
    for mode in ["train", "valid", "test"]:
        data_loader = load.get_data_loader(data_config, mode)
        write_tfrecords(data_dir=os.path.join(tfrecord_dir, mode), data_generator=data_loader.get_generator(),
                        examples_per_tfrecord=examples_per_tfrecord)


if __name__ == "__main__":
    main()
