"""
Module for IO of files in relation to
data loading.
"""
import glob
import itertools as it
import os
import random
from typing import List, Tuple, Union

import h5py


def get_h5_sorted_keys(filename: str) -> List[str]:
    """
    Function to get sorted keys from filename
    :param filename: h5 file.
    :return: sorted keys of h5 file.
    """
    with h5py.File(filename, "r") as h5_file:
        return sorted(h5_file.keys())


def get_sorted_file_paths_in_dir_with_suffix(
    dir_path: str, suffix: Union[str, List[str]]
) -> List[Tuple[str, ...]]:
    """
    Return the path of all files under the given directory.

    :param dir_path: path of the directory
    :param suffix: suffix of file names like h5, nii.gz, nii, should not start with .
    :return: list of relative file path, each element is (file_path, suffix)
    assuming the full path of the file is dir_path/file_path.suffix
    """
    if isinstance(suffix, str):
        suffix = [suffix]
    paths = []
    for suffix_i in suffix:
        # full_path is dir_path/file_path.suffix
        full_paths = glob.glob(
            os.path.join(dir_path, "**", "*." + suffix_i), recursive=True
        )
        file_paths = [
            os.path.relpath(path=p, start=dir_path)[: -(len(suffix_i) + 1)]
            for p in full_paths
        ]
        paths += [(p, suffix_i) for p in file_paths]
    return sorted(paths)


def check_difference_between_two_lists(list1: list, list2: list, name: str):
    """
    Raise error if two lists are not identical

    :param list1: list
    :param list2: list
    :param name: name to be printed in case of difference
    """
    diff = [(x, y) for x, y in it.zip_longest(list1, list2) if x != y]
    if len(diff) > 0:
        raise ValueError(f"{name} are not identical\n" f"difference are {diff}\n")


def get_label_indices(num_labels: int, sample_label: str) -> list:
    """
    Function to get sample label indices for a given number
    of labels and a sampling policy
    :param num_labels: int number of labels
    :param sample_label: method for sampling the labels
    :return: list of labels defined by the sampling method.
    """
    if sample_label == "sample":  # sample a random label
        return [random.randrange(num_labels)]
    elif sample_label == "all":  # use all labels
        return list(range(num_labels))
    else:
        raise ValueError("Unknown label sampling policy %s" % sample_label)
