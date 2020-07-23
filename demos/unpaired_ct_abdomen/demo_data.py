import tarfile
import gzip
import os
import shutil

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

# 2. Extract all images in "Training/img"
for gz_image_file in os.listdir(img_folder_name):
    with gzip.open(os.path.join(img_folder_name, gz_image_file)) as gzip_file: 
        image_data = gzip_file.read()
    nii_image_file = open(os.path.join(img_folder_name, gz_image_file[:-3]),"wb")
    nii_image_file.write(image_data)
    nii_image_file.close
    os.remove(os.path.join(img_folder_name, gz_image_file)) #it should check that the file actually exists

# 3. Extract all labels in "Training/label"
for gz_label_file in os.listdir(label_folder_name):
    with gzip.open(os.path.join(label_folder_name, gz_label_file)) as gzip_file: 
        label_data = gzip_file.read()
    nii_label_file = open(os.path.join(label_folder_name, gz_label_file[:-3]),"wb")
    nii_label_file.write(label_data)
    nii_label_file.close
    os.remove(os.path.join(label_folder_name, gz_label_file)) #it should check that the file actually exists

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

validation_img_files = img_files[int(num_cases*test_split):int(num_cases*test_split)+int(num_cases*validation_split)]
validation_label_files = label_files[int(num_cases*test_split):int(num_cases*test_split)+int(num_cases*validation_split)]
print("The following files will be used in validation: ")
print(validation_img_files)

train_img_files = img_files[int(num_cases*test_split)+int(num_cases*validation_split):]
train_label_files = label_files[int(num_cases*test_split)+int(num_cases*validation_split):]
print("The following files will be used in training: ")
print(train_img_files)

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

# 7.- Copy data into test directory
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