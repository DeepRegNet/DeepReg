import os

import tensorflow as tf

import deepreg.config.parser as config_parser
import deepreg.data.load as load
from deepreg.data.tfrecord import write_tfrecords, parser

if __name__ == "__main__":
    config_path = "deepreg/config/mr_us_ddf.yaml"
    config = config_parser.load(config_path)
    data_config = config["data"]
    for mode in ["train", "test"]:
        data_loader = load.get_data_loader(data_config, mode)
        write_tfrecords(data_dir="data/mr_us/tfrecords/%s/" % mode, data_generator=data_loader.get_generator())

    TF_RECORDS_COMPRESSION_TYPE = "GZIP"
    data_dir = "data/mr_us/tfrecords/train/"
    filenames = [data_dir + x for x in os.listdir(data_dir) if x.endswith(".tfrecords")]
    dataset = tf.data.TFRecordDataset(filenames, compression_type=TF_RECORDS_COMPRESSION_TYPE)
    dataset = dataset.map(parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    for example in dataset.take(1):
        (moving_image, fixed_image, moving_label, indices), fixed_label = example
        print(moving_image.shape, fixed_image.shape, moving_label.shape, indices.shape, fixed_label.shape)
