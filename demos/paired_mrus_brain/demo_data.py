import os
import shutil
import zipfile

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import get_file
from tqdm import tqdm

DOWNLOAD_FULL_DATA = False
DATA_PATH = "dataset"
main_path = os.getcwd()

project_dir = os.path.join(main_path, r"demos/paired_mrus_brain")
os.chdir(project_dir)

######## PARTIAL PREPROCESSED DATA DOWNLOAD (COMMENT OUT) ########
# Please comment out this code block if full data needs to be used
url = "https://github.com/ucl-candi/dataset_resect/archive/master.zip"
fname = "dataset.zip"
get_file(os.path.join(os.getcwd(), fname), url)

# unzip to a temporary folder
tmp_folder = "dataset_tmp"
with zipfile.ZipFile(fname, "r") as zip_ref:
    zip_ref.extractall(tmp_folder)

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)

# move needed data
shutil.move(
    os.path.join(tmp_folder, "dataset_resect-master", "paired_mr_us_brain", "test"),
    os.path.join("dataset", "test"),
)
shutil.move(
    os.path.join(tmp_folder, "dataset_resect-master", "paired_mr_us_brain", "train"),
    os.path.join("dataset", "train"),
)

# remove temporary folder
os.remove(fname)
shutil.rmtree(tmp_folder)

######## DOWNLOAD MODEL CKPT FROM MODEL ZOO ########
url = "https://github.com/DeepRegNet/deepreg-model-zoo/raw/master/paired_mrus_brain_demo_logs.zip"
fname = "pretrained.zip"
get_file(os.path.join(os.getcwd(), fname), url)

with zipfile.ZipFile(fname, "r") as zip_ref:
    zip_ref.extractall(os.path.join("dataset", "pretrained"))

# remove pretrained.zip
os.remove(fname)

# download full data
if not DOWNLOAD_FULL_DATA:
    exit()
print("Code for downloading full data is not tested.")

if os.path.exists("dataset_resect") is not True:
    os.mkdir("dataset_resect")
    os.mkdir(r"dataset_resect/paired_mr_us_brain")
url = "https://ns9999k.webs.sigma2.no/10.11582_2020.00025/EASY-RESECT.zip"
fname = "EASY-RESECT.zip"
path_to_zip_file = "dataset_resect"
get_file(os.path.join(os.getcwd(), path_to_zip_file, fname), url)
with zipfile.ZipFile(os.path.join(path_to_zip_file, fname), "r") as zip_ref:
    zip_ref.extractall(os.path.join(path_to_zip_file, "paired_mr_us_brain"))
path_to_nifti = os.path.join(
    path_to_zip_file, "paired_mr_us_brain", "EASY-RESECT", "NIFTI"
)
all_folders = os.listdir(path_to_nifti)
for folder in all_folders:
    source = os.path.join(path_to_nifti, folder)
    destination = "dataset_resect/paired_mr_us_brain"
    shutil.move(source, destination)
print("Files restructured!")
test_ratio = 0.25
path_to_data = "dataset_resect/paired_mr_us_brain"
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
                    img = nib.Nifti1Image(arr, affine=np.eye(4))
                    img.to_filename(
                        os.path.join(
                            path_to_data,
                            folder,
                            sub_folder,
                            file.split(".nii")[0] + "_resized.nii.gz",
                        )
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
shutil.rmtree(r"dataset_resect/paired_mr_us_brain/train/EASY-RESECT")
shutil.rmtree(r"dataset_resect/paired_mr_us_brain/train/__MACOSX")

# Preprocess the downloaded data
if os.path.exists("dataset_resect/README.md"):
    os.remove("dataset_resect/README.md")

data_folder = "dataset_resect/paired_mr_us_brain"
folders = os.listdir(os.path.join(project_dir, data_folder))

# Move files into correct directories
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

print("Files moved into correct directories")

# Remove unused files
for folder in folders:
    sub_folders = os.listdir(os.path.join(project_dir, data_folder, folder))
    for sub_folder in sub_folders:
        if "Case" in sub_folder:
            shutil.rmtree(os.path.join(project_dir, data_folder, folder, sub_folder))
print("Unused files removed")

# Rename files to match names
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
            im = nib.load(source)
            nib.save(im, destination)
            os.remove(source)
print("files renamed to match each other")

# Rescale images
c = 0
for folder in folders:
    sub_folders = os.listdir(os.path.join(project_dir, data_folder, folder))
    for sub_folder in sub_folders:
        files = os.listdir(os.path.join(project_dir, data_folder, folder, sub_folder))
        for file in files:
            try:
                if "fixed" in sub_folder:
                    im_data = np.asarray(
                        nib.load(
                            os.path.join(
                                project_dir, data_folder, folder, sub_folder, file
                            )
                        ).dataobj,
                        dtype=np.float32,
                    )

                    im_data = ((im_data + 150) / (1700 + 150)) * 255.0  # rescale image

                    img = nib.Nifti1Image(im_data, affine=None)
                    nib.save(
                        img,
                        os.path.join(
                            project_dir, data_folder, folder, sub_folder, file
                        ),
                    )
                    img = nib.load(
                        os.path.join(
                            project_dir, data_folder, folder, "moving_images", file
                        )
                    )
            except nib.filebasedimages.ImageFileError:
                os.remove(
                    os.path.join(project_dir, data_folder, folder, "fixed_images", file)
                )
                os.remove(
                    os.path.join(
                        project_dir, data_folder, folder, "moving_images", file
                    )
                )
                c = c + 1
print("Images rescaled")
print("All done!")
print("Number of files removed due to not loading properly:", c)
