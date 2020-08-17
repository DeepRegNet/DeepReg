import tarfile
import os
import sys
import shutil
import nibabel as nib
import numpy as np
import wget

# 1.- Create directory
project_dir = r"demos/unpaired_ct_abdomen"
data_folder = os.path.join(project_dir, "dataset")
data_file = os.path.join(data_folder, "L2R_Task3_AbdominalCT.tar") #need to be changed to settings or similar
download = True

if os.path.exists(data_folder):
    if os.path.exists(data_file):
        valid = {"Y": True, "Yes": True, "y": True, "yes": True, "N": False, "No": False, "n": False, "no": False,}
        response = input("Data already exists, download again? [Y/N]: ")
        
        if response in valid.keys():
            download = valid[response]
        else:
            print("Invalid answer. Please try again.")
            sys.exit(1)
else:
    os.mkdir(data_folder)

# 2.- Download the data
if download == True:
    if os.path.exists(data_file):
        os.remove(data_file)
    res = os.system("wget --load-cookies /tmp/cookies.txt -O demos/unpaired_ct_abdomen/dataset/L2R_Task3_AbdominalCT.tar \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1aWyS_mQ5n7X2bTk9etHrn5di2-EZEzyO\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/"+chr(92)+"1"+chr(92)+"n/p\')&id=1aWyS_mQ5n7X2bTk9etHrn5di2-EZEzyO\"")
    if not res == 0:
        os.remove(data_file) # download failed, remove partly downloaded file

if not os.path.exists(data_folder):
    print("There was a problem downloading the data. Please try again.")
    sys.exit(1)

# 3.- Extract data --> This will create a new temporal folder "Training"
if os.path.exists(os.path.join(data_folder,"Training")):
    shutil.rmtree(os.path.join(data_folder,"Training")) # delete temporal folder

tar_file = tarfile.open(data_file)
tar_file.extractall(data_folder)
tar_file.close
img_folder_name = os.path.join(data_folder,"Training/img/")
label_folder_name = os.path.join(data_folder,"Training/label/")

# 4. Normalise and rename all image files
for gz_image_file in os.listdir(img_folder_name):
    image_data = np.asarray(nib.load(os.path.join(img_folder_name, gz_image_file)).dataobj, dtype=np.float32)
    
    if np.min(image_data) < 0 or np.max(255.0) > 1:
        image_data = (image_data-np.min(image_data)) * 255/(np.max(image_data)-np.min(image_data))
        image_data = image_data * 0.9999 # Workaround to avoid error with floats being > 1.0
        nii_image = nib.Nifti1Image(image_data, affine=None)
        nib.save(nii_image, os.path.join(img_folder_name, gz_image_file))
    os.rename(os.path.join(img_folder_name, gz_image_file), os.path.join(img_folder_name, gz_image_file[3:]))

# 5. Normalise and rename all label files, and separate labels in multiple channels and binary
for gz_label_file in os.listdir(label_folder_name):
    label_data = np.asarray(nib.load(os.path.join(label_folder_name, gz_label_file)).dataobj, dtype=np.float32)
    # There are 13 labels in the dataset, and each label has to be in a separate channel. 0 is background.
    masks = np.concatenate([np.expand_dims((label_data == label).astype(np.float32), axis=3) for label in range(1,13)], axis=3)
    masks = masks * 0.9999 # Workaround to avoid error with floats being > 1.0
    nii_labels = nib.Nifti1Image(masks, affine=None)
    nib.save(nii_labels, os.path.join(label_folder_name, gz_label_file))
    os.rename(os.path.join(label_folder_name, gz_label_file), os.path.join(label_folder_name, gz_label_file[5:]))

# 6.- Divide data in training, validation and testing
validation_split = 0.15 # 15% of the data for validation
test_split = 0.07 # 5% of the data for testing

img_files = os.listdir(img_folder_name)
label_files = os.listdir(label_folder_name)

if len(img_files) != len(label_files):
    print("Error. The number of images and labels in " + img_folder_name + " and " +
     label_folder_name + " seem to be different. Plase try again.")
    sys.exit(1)

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

# 7.- Copy data into train folder
train_folder = os.path.join(data_folder, "train")
if os.path.exists(train_folder):
    shutil.rmtree(train_folder) # delete old data

if os.path.exists(train_folder) is not True:
    os.mkdir(train_folder)
    os.mkdir(os.path.join(train_folder, "images"))
    os.mkdir(os.path.join(train_folder, "labels"))
    
for nii_file in train_img_files:
    shutil.move(os.path.join(img_folder_name,nii_file),os.path.join(train_folder,"images",nii_file))
for label_file in train_label_files:
    shutil.move(os.path.join(label_folder_name,label_file),os.path.join(train_folder,"labels",label_file))

# 8.- Copy data into validation folder
valid_folder = os.path.join(data_folder, "valid")
if os.path.exists(valid_folder):
    shutil.rmtree(valid_folder) # delete old data

if os.path.exists(valid_folder) is not True:
    os.mkdir(valid_folder)
    os.mkdir(os.path.join(valid_folder, "images"))
    os.mkdir(os.path.join(valid_folder, "labels"))
    
for nii_file in validation_img_files:
    shutil.move(os.path.join(img_folder_name,nii_file),os.path.join(valid_folder,"images",nii_file))
for label_file in validation_label_files:
    shutil.move(os.path.join(label_folder_name,label_file),os.path.join(valid_folder,"labels",label_file))

# 9.- Copy data into test folder
test_folder = os.path.join(data_folder, "test")
if os.path.exists(test_folder):
    shutil.rmtree(test_folder) # delete old data

if os.path.exists(test_folder) is not True:
    os.mkdir(test_folder)
    os.mkdir(os.path.join(test_folder, "images"))
    os.mkdir(os.path.join(test_folder, "labels"))
    
for nii_file in test_img_files:
    shutil.move(os.path.join(img_folder_name,nii_file),os.path.join(test_folder,"images",nii_file))
for label_file in test_label_files:
    shutil.move(os.path.join(label_folder_name,label_file),os.path.join(test_folder,"labels",label_file))

shutil.rmtree(os.path.join(data_folder,"Training")) # delete temporal folder