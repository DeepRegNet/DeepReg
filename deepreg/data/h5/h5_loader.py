'''
Loads h5 files and some associated information
'''
import os

import h5py
import numpy as np

class H5FileLoader:
    '''
    Loads h5 files and an be used to get various attributes of the files
    Also supports grouped data
    '''
    def __init__(self, dir_path: str, grouped: bool):
        '''
        Generalised loader for h5 files
        
        :param dir_path: path to directory which contains data
        :param grouped: True/ False indicating if data is grouped
        '''
        self.dir_path = dir_path
        self.grouped = grouped
        self.data_dict = None
        
        if grouped:
            self.group_names = sorted(os.listdir(dir_path))
            self.num_groups_in_dir = len(os.listdir(self.dir_path))
            
            self.group_paths = sorted(os.listdir(dir_path))
            for i in range(len(self.group_paths)):
                self.group_paths[i] = os.path.join(self.dir_path, 
                                                   self.group_paths[i])
            

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
    
    def get_data_names(self, fname=None):
        ''' 
        Return list of the names of data in the h5 file
        '''
        if fname is None:
            self.dict_from_h5(fname='data.h5')
        else:
            self.dict_from_h5(fname=fname)
        return list(self.data_dict.keys())
    
    def get_data(self, index: (int, tuple)):
        '''
        Get one data array by specifying an index
        
        :param index: the data index which is required
        
        :returns arr: the data array at the specified index
        '''
        dir_path_orig = self.dir_path
        if isinstance(index, int):
            assert not self.grouped
            assert 0 <= index < len(self.data_dict)
            self.dict_from_h5()
            data_name = list(self.data_dict.keys())[index]
        elif isinstance(index, tuple):
            assert self.grouped
            group_index, sample_index = index
            group_name = self.group_names[group_index]
            self.dir_path = os.path.join(dir_path_orig, group_name)
            self.dict_from_h5(fname=group_name+'.h5')
            data_name = list(self.data_dict.keys())[sample_index]
            self.dir_path = dir_path_orig
            
            

        arr = self.data_dict[data_name]
        if len(arr.shape) == 4 and arr.shape[3] == 1:
            # for labels, if there's only one label, remove the last dimension
            arr = arr[:, :, :, 0]
        return arr
        
    def get_data_names_grouped(self):
        '''
        for grouped folder structure return the names of samples
        :returns: sorted list of all sample names
        '''
        dir_path_orig = self.dir_path
        all_sample_names = []
        for group_name in self.group_names:
            self.dir_path = os.path.join(dir_path_orig, group_name)
            self.dict_from_h5(fname=group_name+'.h5')
            samples_in_group = sorted(self.get_data_names(fname=
                                                          group_name+'.h5'))
            all_sample_names = all_sample_names + samples_in_group
            self.dir_path = dir_path_orig
        return sorted(all_sample_names)
        
    def get_num_images_per_group(self):
        '''
        for grouped folder structure, reurn the number of images in each group
        :returns: list of numer of images per group
        '''
        dir_path_orig = self.dir_path
        num_images_per_group = []
        for group_name in self.group_names:
            self.dir_path = os.path.join(dir_path_orig, group_name)
            self.dict_from_h5(fname=group_name+'.h5')
            samples_in_group = sorted(self.get_data_names(fname=
                                                          group_name+'.h5'))
            num_samples = len(samples_in_group)
            num_images_per_group.append(num_samples)
            self.dir_path = dir_path_orig
        return num_images_per_group

