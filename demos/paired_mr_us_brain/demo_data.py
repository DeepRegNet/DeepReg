import os
import shutil
import zipfile

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import get_file
from tqdm import tqdm

main_path = os.getcwd()

project_dir = os.path.join(main_path, r"demos/paired_mr_us_brain")
os.chdir(project_dir)

######## ########

if os.path.exists("dataset_respect") is not True:
    os.mkdir("dataset_respect")
    os.mkdir(r"dataset_respect/paired_mr_us_brain")
url = "https://ns9999k.webs.sigma2.no/10.11582_2020.00025/EASY-RESECT.zip"
fname = "EASY-RESECT.zip"
path_to_zip_file = r"dataset_respect"
get_file(os.path.join(os.getcwd(), path_to_zip_file, fname), url)
with zipfile.ZipFile(os.path.join(path_to_zip_file, fname), "r") as zip_ref:
    zip_ref.extractall(os.path.join(path_to_zip_file, "paired_mr_us_brain"))
path_to_nifti = os.path.join(
    path_to_zip_file, "paired_mr_us_brain", "EASY-RESECT", "NIFTI"
)
all_folders = os.listdir(path_to_nifti)
for folder in all_folders:
    source = os.path.join(path_to_nifti, folder)
    destination = r"dataset_respect/paired_mr_us_brain"
    shutil.move(source, destination)
print("Files restructured!")
test_ratio = 0.25
path_to_data = r"dataset_respect/paired_mr_us_brain"
cases_list = os.listdir(path_to_data)
os.mkdir(os.path.join(path_to_data, "test"))
os.mkdir(os.path.join(path_to_data, "train"))
num_test = round(len(cases_list) * test_ratio)
for folder in cases_list[:num_test]:
    source = os.path.join(path_to_data, folder)
    destination = os.path.join(path_to_data, "test")
    shutil.move(source, destination)
for folder in cases_list[num_test:]:
    source = os.path.join(path_to_data, folder)
    destination = os.path.join(path_to_data, "train")
    shutil.move(source, destination)
folders = os.listdir(path_to_data)
for folder in folders:
    sub_folders = os.listdir(os.path.join(path_to_data, folder))
    for sub_folder in tqdm(sub_folders):
        if "DS_St" in sub_folder:
            os.remove(os.path.join(path_to_data, folder, sub_folder))
        else:
            files = os.listdir(os.path.join(path_to_data, folder, sub_folder))
        for file in files:
            if "T1" in file:
                arr = nib.load(
                    os.path.join(path_to_data, folder, sub_folder, file)
                ).get_data()
                resize_arr = np.resize(arr, (64, 64, 72))
                img = nib.Nifti1Image(resize_arr, None)
                nib.save(
                    img,
                    os.path.join(
                        path_to_data,
                        folder,
                        sub_folder,
                        file.split(".nii")[0] + "_resized.nii.gz",
                    ),
                )
            elif "US" in file:
                img = nib.load(os.path.join(path_to_data, folder, sub_folder, file))
                nib.save(
                    img,
                    os.path.join(
                        path_to_data,
                        folder,
                        sub_folder,
                        file.split(".ni")[0] + ".nii.gz",
                    ),
                )


######## ########

# os.system("git clone https://github.com/ucl-candi/dataset_respect.git")

######## ########


if os.path.exists("dataset_respect/README.md"):
    os.remove("dataset_respect/README.md")

data_folder = "dataset_respect/paired_mr_us_brain"

folders = os.listdir(os.path.join(project_dir, data_folder))

######## MOVE CORRECT FILES INTO CORRECT DIRECTORIES ########

for folder in folders:
    sub_folders = os.listdir(os.path.join(project_dir, data_folder, folder))

    if (
        os.path.exists(os.path.join(project_dir, data_folder, folder, "fixed_images"))
        is not True
    ):
        os.mkdir(os.path.join(project_dir, data_folder, folder, "fixed_images"))
        os.mkdir(os.path.join(project_dir, data_folder, folder, "moving_images"))
    for sub_folder in sub_folders:
        files = os.listdir(os.path.join(project_dir, data_folder, folder, sub_folder))
        for file in files:
            if "T1" in file:
                source = os.path.join(
                    project_dir, data_folder, folder, sub_folder, file
                )
                destination = os.path.join(
                    project_dir, data_folder, folder, "fixed_images", file
                )
                shutil.move(source, destination)
            elif "US" in file:
                source = os.path.join(
                    project_dir, data_folder, folder, sub_folder, file
                )
                destination = os.path.join(
                    project_dir, data_folder, folder, "moving_images", file
                )
                shutil.move(source, destination)


######## REMOVE FILES THAT WILL NOT BE USED ########

for folder in folders:
    sub_folders = os.listdir(os.path.join(project_dir, data_folder, folder))
    for sub_folder in sub_folders:
        if "Case" in sub_folder:
            shutil.rmtree(os.path.join(project_dir, data_folder, folder, sub_folder))


######## RENAME FILES TO MATCH NAMES ########

for folder in folders:
    sub_folders = os.listdir(os.path.join(project_dir, data_folder, folder))
    for sub_folder in sub_folders:
        files = os.listdir(os.path.join(project_dir, data_folder, folder, sub_folder))
        for file in files:
            source = os.path.join(project_dir, data_folder, folder, sub_folder, file)
            destination = os.path.join(
                project_dir,
                data_folder,
                folder,
                sub_folder,
                file.split("-")[0] + ".nii.gz",
            )
            os.rename(source, destination)

######## RESCALE THE IMAGES ########

for folder in folders:
    sub_folders = os.listdir(os.path.join(project_dir, data_folder, folder))
    for sub_folder in sub_folders:
        files = os.listdir(os.path.join(project_dir, data_folder, folder, sub_folder))
        for file in files:
            if "fixed" in sub_folder:
                im_data = np.asarray(
                    nib.load(
                        os.path.join(project_dir, data_folder, folder, sub_folder, file)
                    ).dataobj,
                    dtype=np.float32,
                )

                im_data = ((im_data + 150) / (1200 + 150)) * 255.0  # rescale image

                img = nib.Nifti1Image(im_data, affine=None)
                nib.save(
                    img,
                    os.path.join(project_dir, data_folder, folder, sub_folder, file),
                )
