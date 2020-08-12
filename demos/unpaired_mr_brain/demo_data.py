import os
import shutil
import tarfile
import zipfile
from os import listdir, makedirs, remove, system
from os.path import exists, join

import nibabel as nib
import numpy as np

##############
# Parameters #
##############

data_splits = ["train", "test"]
num_labels = 3

# Main project directory
main_path = os.getcwd()
os.chdir(main_path)

# Demo directory
project_dir = r"demos/unpaired_mr_brain"
os.chdir(join(main_path, project_dir))

# Data storage directory
data_folder_name = "data"
path_to_data_folder = join(main_path, project_dir, data_folder_name)

# Pretrained model storage directory
model_folder_name = "logs"
path_to_model_folder = join(main_path, model_folder_name)

#################
# Download data #
#################
# Data
FNAME = "data_mr_brain.tar"
url = (
    "'https://drive.google.com/uc?export=download&id=1RvJIjG2loU8uGkWzUuGjqVcGQW2RzNYA'"
)
system("wget " + " -O " + FNAME + " " + url)
print("The file ", FNAME, " has successfully been downloaded!")

if exists(path_to_data_folder) is not True:
    makedirs(path_to_data_folder)

with tarfile.open(join(main_path, project_dir, FNAME), "r") as tar_ref:
    tar_ref.extractall(data_folder_name)

remove(FNAME)
print("Files unzipped!")


# Model
pretrained_model = "unpaired_mr_brain.zip"
url_model = (
    "https://github.com/DeepRegNet/deepreg-model-zoo/raw/master/" + pretrained_model
)

system("wget " + " -O " + pretrained_model + " " + url_model)
print("The file ", pretrained_model, " has successfully been downloaded!")

if exists(path_to_model_folder) is not True:
    makedirs(path_to_model_folder)

with zipfile.ZipFile(join(main_path, project_dir, pretrained_model), "r") as zip_ref:
    zip_ref.extractall(path_to_model_folder)

remove(pretrained_model)


##################
# Create dataset #
##################
path_to_init_img = join(path_to_data_folder, "Training", "img")
path_to_init_label = join(path_to_data_folder, "Training", "label")

path_to_train = join(path_to_data_folder, "train")
path_to_test = join(path_to_data_folder, "test")

if not exists(path_to_train):
    makedirs(join(path_to_train, "images"))
    makedirs(join(path_to_train, "labels"))
    makedirs(join(path_to_train, "masks"))
else:
    shutil.rmtree(path_to_train)
    makedirs(join(path_to_train, "images"))
    makedirs(join(path_to_train, "labels"))
    makedirs(join(path_to_train, "masks"))

if not exists(path_to_test):
    makedirs(join(path_to_test, "images"))
    makedirs(join(path_to_test, "labels"))
    makedirs(join(path_to_test, "masks"))
    shutil.rmtree(path_to_test)
    makedirs(join(path_to_test, "images"))
    makedirs(join(path_to_test, "labels"))
    makedirs(join(path_to_test, "masks"))

img_files = listdir(path_to_init_img)
for f in img_files:
    num_subject = int(f.split("_")[1].split(".")[0])

    if num_subject < 311:
        shutil.copy(join(path_to_init_img, f), join(path_to_train, "images"))
    else:
        shutil.copy(join(path_to_init_img, f), join(path_to_test, "images"))

img_files = listdir(path_to_init_label)
for f in img_files:
    num_subject = int(f.split("_")[1].split(".")[0])
    if num_subject < 311:
        shutil.copy(join(path_to_init_label, f), join(path_to_train, "labels"))
    else:
        shutil.copy(join(path_to_init_label, f), join(path_to_test, "labels"))

print("Files succesfully copied to " + path_to_train + " and " + path_to_test)


#################
# Preprocessing #
#################
for ds in data_splits:
    path = join(path_to_data_folder, ds, "images")
    files = listdir(path)
    for f in files:
        proxy = nib.load(join(path, f))
        data = np.asarray(proxy.dataobj)
        mask = np.zeros_like(data)
        center = [int(s / 2) for s in data.shape]
        mask_tuple = []
        axes = [2, 0, 1]
        for it_dim in range(len(data.shape)):
            dim = data.shape[it_dim]
            axes = [np.mod(a + 1, 3) for a in axes]
            data_tmp = np.transpose(data, axes=axes)

            it_voxel_init = 0
            values_init = data_tmp[it_voxel_init, center[it_dim]]
            while True:
                it_voxel_init += 1
                values = data_tmp[it_voxel_init, center[it_dim]]
                if np.sum((values - values_init) ** 2) > 0:
                    break

            it_voxel_fi = dim - 1
            values_fi = data_tmp[it_voxel_fi, center[it_dim]]
            while True:
                it_voxel_fi -= 1
                values = data_tmp[it_voxel_fi, center[it_dim]]
                if np.sum((values - values_fi) ** 2) > 1:
                    it_voxel_fi += 1
                    break

            mask_tuple.append((it_voxel_init, it_voxel_fi))

        mask[
            mask_tuple[0][0] : mask_tuple[0][1],
            mask_tuple[1][0] : mask_tuple[1][1],
            mask_tuple[2][0] : mask_tuple[2][1],
        ] = 1
        img = nib.Nifti1Image(mask, affine=proxy.affine)
        nib.save(img, join(path_to_data_folder, ds, "masks", f))

        data = data * mask
        M = np.max(data)
        m = np.min(data)
        if M > 255:
            data = (data - m) / (M - m) * 255.0
        img = nib.Nifti1Image(data, affine=proxy.affine)
        nib.save(img, join(path, f))

print("Images have been correctly normalized between [0, 255]")

# One hot encoding labels labels
for ds in data_splits:
    path = join(path_to_data_folder, ds, "labels")
    files = listdir(path)
    for f in files:
        proxy = nib.load(join(path, f))
        labels = np.asarray(proxy.dataobj)
        labels_one_hot = []
        for it_l in range(1, num_labels):
            index_labels = np.where(labels == it_l)
            mask = np.zeros_like(labels)
            mask[index_labels] = 1
            labels_one_hot.append(mask)
        labels_one_hot = np.stack(labels_one_hot, axis=-1)
        img = nib.Nifti1Image(labels_one_hot, proxy.affine)
        nib.save(img, join(path, f))

print(
    "Labels have been one-hot encoding using a total of " + str(num_labels) + " labels."
)
