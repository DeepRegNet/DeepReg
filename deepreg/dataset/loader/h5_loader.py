"""
Loads h5 files and some associated information
"""
import os

import h5py
import numpy as np

from deepreg.dataset.loader.interface import FileLoader
from deepreg.dataset.util import get_sorted_filenames_in_dir


class H5FileLoader(FileLoader):
    """Generalized loader for h5 files"""

    def __init__(self, dir_path: str, grouped: bool):
        super(H5FileLoader, self).__init__(dir_path=dir_path, grouped=grouped)
        self.file_paths = get_sorted_filenames_in_dir(dir_path=dir_path,
                                                      suffix="h5")
        self.dir_path = '/'.join(dir_path.split('/')[:-1])
        self.fname = dir_path.split('/')[-1] + '.h5'
        self.data_dict = None
        if grouped:
            self.num_groups = self.get_num_groups()

    def dict_from_h5(self, fname):
        """
        Dictionary is generated from data in an h5 file
        
        :param fname: name of the h5 file which stores the data
        """
        with h5py.File(os.path.join(self.dir_path, fname), 'r') as f:
            keys = sorted(f.keys())
            data_dict = {}
            for key in keys:
                data_dict[key] = np.asarray(f[key], dtype=np.float32)
        self.data_dict = data_dict

    def get_data_names(self):
        """
        Return list of the names of data in the h5 file
        """
        self.dict_from_h5(fname=self.fname)
        return list(self.data_dict.keys())

    def get_data(self, index: (int, tuple)):
        """
        Get one data array by specifying an index
        
        :param index: the data index which is required
        
        :returns arr: the data array at the specified index
        """
        if isinstance(index, int):
            assert not self.grouped
            self.dict_from_h5(fname=self.fname)
            assert 0 <= index < len(self.data_dict)
            data_name = list(self.data_dict.keys())[index]
        elif isinstance(index, tuple):
            assert self.grouped
            group_index, sample_index = index
            data_name = 'group-' + str(group_index + 1) + '-' + str(sample_index + 1)
            self.dict_from_h5(fname=self.fname)
        arr = self.data_dict[data_name]
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            arr = arr[:, :, :, 0]
        return arr

    def get_data_names_grouped(self):
        """
        for grouped folder structure return the names of samples
        :returns: sorted list of all sample names
        """
        self.dict_from_h5(fname=self.fname)
        all_sample_names = sorted(self.data_dict.keys())
        return all_sample_names

    def get_data_ids(self):
        '''
        get data names
        :returns: names of data in list format
        '''
        if self.grouped:
            return self.get_data_names_grouped()
        else:
            return self.get_data_names()

    def get_num_images(self) -> int:
        '''
        get number of images in data
        :returns: number of images
        '''
        return len(self.get_data_names())

    def get_num_images_per_group(self):
        '''
        get number of images in each group
        :returns: list of numbers corresponding to number of images per group
        '''
        assert self.grouped
        self.dict_from_h5(fname=self.fname)
        all_images = sorted(self.data_dict.keys())
        group_numbers = []
        for image in all_images:
            group_number = int(image.split('-')[-2])
            group_numbers.append(group_number)
        num_images_per_group = []
        for i in range(self.num_groups):
            num_images = int(len(np.where(np.array(group_numbers) == i + 1)[0] + 1))
            num_images_per_group.append(num_images)
        return num_images_per_group

    def get_num_groups(self) -> int:
        '''
        get number of groups
        :returns: total number of groups
        '''
        assert self.grouped
        self.dict_from_h5(fname=self.fname)
        all_images = sorted(self.data_dict.keys())
        group_numbers = []
        for image in all_images:
            group_number = int(image.split('-')[-2])
            group_numbers.append(group_number)
        num_groups = np.amax(np.array(group_numbers))
        return num_groups
