import tarfile
import os
import shutil
import nibabel as nib
import numpy as np

# 1.- Extract data --> This will create a new folder "Training"
project_dir = r"demos/unpaired_ct_abdomen"
data_folder = os.path.join(project_dir, "dataset")
data_file = os.path.join(data_folder, "L2R_Task3_AbdominalCT.tar") #need to be changed to settings or similar

# TODO if Training folder exists --> eliminate

tar_file = tarfile.open(data_file)
tar_file.extractall(data_folder)
tar_file.close
img_folder_name = os.path.join(data_folder,"Training/img/")
label_folder_name = os.path.join(data_folder,"Training/label/")

# 2. Normalise and rename all image files
for gz_image_file in os.listdir(img_folder_name):
    image_data = np.asarray(nib.load(os.path.join(img_folder_name, gz_image_file)).dataobj, dtype=np.float32)
    
    if np.min(image_data) < 0 or np.max(255.0) > 1:
        image_data = (image_data-np.min(image_data)) * 255/(np.max(image_data)-np.min(image_data))
        nii_image = nib.Nifti1Image(image_data, affine=None)
        nib.save(nii_image, os.path.join(img_folder_name, gz_image_file))
    os.rename(os.path.join(img_folder_name, gz_image_file), os.path.join(img_folder_name, gz_image_file[3:]))

# 3. Normalise and rename all label files, and separate labels in multiple channels and binary
for gz_label_file in os.listdir(label_folder_name):
    label_data = np.asarray(nib.load(os.path.join(label_folder_name, gz_label_file)).dataobj, dtype=np.int)
    #labels = np.delete(np.unique(label_data),0)
    masks = np.concatenate([np.expand_dims((label_data == label).astype(np.int), axis=3) for label in range(1,13)], axis=3)
    nii_labels = nib.Nifti1Image(masks, affine=None)
    nib.save(nii_labels, os.path.join(label_folder_name, gz_label_file))
    os.rename(os.path.join(label_folder_name, gz_label_file), os.path.join(label_folder_name, gz_label_file[5:]))
    print("min = %.7f max = %.7f " % (np.min(masks), np.max(masks)))

# 4.- Divide data in training, validation and testing
validation_split = 0.15 # 15% of the data for validation
test_split = 0.07 # 5% of the data for testing

img_files = os.listdir(img_folder_name)
label_files = os.listdir(label_folder_name)

# TODO error if lenght of img_files != label files

num_cases = len(img_files)

test_img_files = img_files[0:int(num_cases*test_split)]
test_label_files = label_files[0:int(num_cases*test_split)]
print("The following files will be used in testing: ")
print(test_img_files)
print(test_label_files)

validation_img_files = img_files[int(num_cases*test_split):int(num_cases*test_split)+int(num_cases*validation_split)]
validation_label_files = label_files[int(num_cases*test_split):int(num_cases*test_split)+int(num_cases*validation_split)]
print("The following files will be used in validation: ")
print(validation_img_files)
print(validation_label_files)

train_img_files = img_files[int(num_cases*test_split)+int(num_cases*validation_split):]
train_label_files = label_files[int(num_cases*test_split)+int(num_cases*validation_split):]
print("The following files will be used in training: ")
print(train_img_files)
print(train_label_files)

# 5.- Copy data into train folder
train_folder = os.path.join(data_folder, "train")

if os.path.exists(train_folder) is not True:
    os.mkdir(train_folder)
    os.mkdir(os.path.join(train_folder, "images"))
    os.mkdir(os.path.join(train_folder, "labels"))
    
for nii_file in train_img_files:
    shutil.move(os.path.join(img_folder_name,nii_file),os.path.join(train_folder,"images",nii_file))
for label_file in train_label_files:
    shutil.move(os.path.join(label_folder_name,label_file),os.path.join(train_folder,"labels",label_file))

# 6.- Copy data into validation folder
valid_folder = os.path.join(data_folder, "valid")

if os.path.exists(valid_folder) is not True:
    os.mkdir(valid_folder)
    os.mkdir(os.path.join(valid_folder, "images"))
    os.mkdir(os.path.join(valid_folder, "labels"))
    
for nii_file in validation_img_files:
    shutil.move(os.path.join(img_folder_name,nii_file),os.path.join(valid_folder,"images",nii_file))
for label_file in validation_label_files:
    shutil.move(os.path.join(label_folder_name,label_file),os.path.join(valid_folder,"labels",label_file))

# 7.- Copy data into test folder
test_folder = os.path.join(data_folder, "test")

if os.path.exists(test_folder) is not True:
    os.mkdir(test_folder)
    os.mkdir(os.path.join(test_folder, "images"))
    os.mkdir(os.path.join(test_folder, "labels"))
    
for nii_file in test_img_files:
    shutil.move(os.path.join(img_folder_name,nii_file),os.path.join(test_folder,"images",nii_file))
for label_file in test_label_files:
    shutil.move(os.path.join(label_folder_name,label_file),os.path.join(test_folder,"labels",label_file))

shutil.rmtree(os.path.join(data_folder,"Training"))