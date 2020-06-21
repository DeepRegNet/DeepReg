'''
Tests functinality of the H5FileLOader
'''

import numpy as np

from deepreg.data.h5.util import H5FileLoader

class Test():
    '''
    Tests the funcitonality of the H5FileLoader
    The test_xyz methods will return True is the test is passed
    '''
    @staticmethod
    def check_list_equal(list1, list2):
        '''
        given two lists, return True if equal and False if not
        :param list1:
        :param list2:
        :return:
        '''
        return list1.sort() == list2.sort()
 
    @staticmethod
    def get_shapes_for_dict(dictionary):
        '''
        given a dictionary of numpy arrays return shapes of arrays
        :param dicitonary: dictionary of numpy arrays
        :return: list of shapes of numpy arrays in sorted order
        '''
        shapes = []
        keys = sorted(dictionary.keys())
        for key in keys:
            shape = np.shape(dictionary[key])
            shapes.append(shape)
        return shapes
    
    @staticmethod
    def check_tuple_of_int_equal(tuple1, tuple2):
        '''
        given two tuples, return True if equal and False if not
        :param tuple1:
        :param tuple2:
        :return:
        '''
        test = True
        for i in range(len(tuple1)):
            test = test and int(tuple1[i]) == int(tuple2[i])
        return test
    
    def test_dict_from_h5(self):
        '''
        check if the dict_from_h5 function returns the expected keys
        and shapes of arrays when loading in test data
        :return: True/ False
        '''
        filename = 'data.h5'
        directory = 'data/h5_mr_us/mr_us/paired/test/fixed_images'
        
        loader = H5FileLoader(directory)
        loader.dict_from_h5(fname=filename)
        obtained = loader.data_dict
        obtained_keys = sorted(obtained.keys())
        obtained_shapes = self.get_shapes_for_dict(obtained)
        
        expected_keys = ['case000025.nii.gz']
        expected_shapes = [(44, 59, 41)]
        
        test_1 = self.check_list_equal(obtained_keys, expected_keys)
        test_2 = self.check_tuple_of_int_equal(obtained_shapes[0],
                                               expected_shapes[0])
        
        return test_1 and test_2
        
    def test_get_data_names(self):
        '''
        check if the get_data_names function returns the expected keys
        :return: True/ False
        '''
        directory = 'data/h5_mr_us/mr_us/paired/test/fixed_images'

        loader = H5FileLoader(directory)
        
        obtained = loader.get_data_names()
        expected = ['case000025.nii.gz']    

        test = self.check_list_equal(obtained, expected)
        return test

    def test_get_data(self):
        '''
        check if the get_data function returns the 
        shapes of arrays when loading in test data
        :return: True/ False
        '''
        directory = 'data/h5_mr_us/mr_us/paired/test/fixed_images'
        
        loader = H5FileLoader(directory)
        
        obtained = np.shape(loader.get_data(index=0))
        
        expected = (44, 59, 41)
        
        test = self.check_tuple_of_int_equal(obtained, expected)
        return test
