import tarfile
import gzip
import os

# 1.- Extract data --> This will create a new folder "Training"
data_file = "data/L2R_Task3_AbdominalCT.tar"
data_folder_name = "data"

tar_file = tarfile.open(data_file)
tar_file.extractall(data_folder_name)
tar_file.close
img_folder_name = os.path.join(data_folder_name,"Training/img/")
label_folder_name = os.path.join(data_folder_name,"Training/label/")

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

# 2.- Sort data in demo directory
#project_dir = r"demos/unpaired_ct_abdomen"
#os.chdir(project_dir)

# 3.- Copy data into train directory
#path_to_train = os.path.join(main_path, project_dir, data_folder_name, "train")

# 4.- Copy data into validation directory

# 5.- Copy data into test directory



