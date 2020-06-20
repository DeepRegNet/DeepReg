import os
import h5py
import nibabel as nib
import numpy as np
import hdfdict


# NOTE: Before running the script please create a copy of the data/mr_us directory
# then change the maindir variable which will be the path to the copied directory location 
# for exmple maindir ends with /mr_us
# RUNNING THIS SCRIPT WILL RESULT IN THE DELETION OF NIFTI FILES IN THE COPIED DATA DIRECTORY
maindir = r''
os.chdir(maindir)
folders = os.listdir()



# make data.h5 files for all sub folders in paired
paired_dir = os.path.join(maindir, 'paired')
folders = os.listdir(paired_dir)

for folder in folders:
    folder_dir = os.path.join(paired_dir,folder)
    subfolders = os.listdir(folder_dir)
    for subfolder in subfolders:
        subfolder_dir = os.path.join(folder_dir,subfolder)
        files = os.listdir(subfolder_dir)
        os.chdir(subfolder_dir)
        dictionary = {}
        for file in files:
            if file.endswith('gz'):
                data = np.asarray(nib.load(file).dataobj, dtype=np.float32)
                name = file
                dictionary[name]=data
        hdfdict.dump(dictionary, 'data.h5')
            


# same for unpaired
unpaired_dir = os.path.join(maindir, 'unpaired')
folders = os.listdir(unpaired_dir)

for folder in folders:
    folder_dir = os.path.join(unpaired_dir,folder)
    subfolders = os.listdir(folder_dir)
    for subfolder in subfolders:
        subfolder_dir = os.path.join(folder_dir,subfolder)
        files = os.listdir(subfolder_dir)
        os.chdir(subfolder_dir)
        dictionary = {}
        for file in files:
            if file.endswith('gz'):
                data = np.asarray(nib.load(file).dataobj, dtype=np.float32)
                name = file
                dictionary[name]=data
        hdfdict.dump(dictionary, 'data.h5')


# delete nifti images from unpaired
unpaired_dir = os.path.join(maindir, 'unpaired')
folders = os.listdir(unpaired_dir)
for folder in folders:
    folder_dir = os.path.join(unpaired_dir,folder)
    subfolders = os.listdir(folder_dir)
    for subfolder in subfolders:
        subfolder_dir = os.path.join(folder_dir,subfolder)
        files = os.listdir(subfolder_dir)
        os.chdir(subfolder_dir)
        for file in files:
            if file.endswith('gz'):
                os.remove(file)


# delete nifti images from paired
paired_dir = os.path.join(maindir, 'paired')
folders = os.listdir(paired_dir)
for folder in folders:
    folder_dir = os.path.join(paired_dir,folder)
    subfolders = os.listdir(folder_dir)
    for subfolder in subfolders:
        subfolder_dir = os.path.join(folder_dir,subfolder)
        files = os.listdir(subfolder_dir)
        os.chdir(subfolder_dir)
        for file in files:
            if file.endswith('gz'):
                os.remove(file)



