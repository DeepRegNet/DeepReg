'''
Loads h5 files
'''
import os

import numpy as np

import h5py


class H5FileLoader:
    '''
    General loader for h5 files
    Supports:
    loading an array from h5 data
    converting h5 data into a dictionary 
    getting the keys from the h5 file
    '''
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.data_dict = None

    def dict_from_h5(self, fname='data.h5'):
        '''
        Dictionary is generated from data in an h5 file
        
        :param fname: name of the h5 file which stores the data
        '''
        with h5py.File(os.path.join(self.dir_path, fname), 'r') as f:
            keys = sorted(f.keys())
            data_dict = {}
            for key in keys:
                data_dict[key] = np.asarray(f[key], dtype=np.float32)
        self.data_dict = data_dict
    
    def get_data_names(self):
        ''' 
        Return list of the names of data in the h5 file
        '''
        self.dict_from_h5()
        return list(self.data_dict.keys())
    
    def get_data(self, index: int):
        '''
        Get one data array by specifying an index
        
        :param index: the data index which is required
        
        :returns arr: the data array at the specified index
        '''
        self.dict_from_h5()
        assert 0 <= index < len(self.data_dict)
        arr = self.data_dict[list(self.data_dict.keys())[index]]
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            arr = arr[:, :, :, 0]
        return arr
