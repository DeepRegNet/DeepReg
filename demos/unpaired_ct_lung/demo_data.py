import os
import random
import shutil
import zipfile

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import get_file
from tqdm import tqdm

# if wget is installed remove the following line from comment
# import wget

# if already in the abc/DeepReg directory then do nothing, otherwise
# use os.chdir(r'abc/DeepReg') before this line
main_path = os.getcwd()
os.chdir(main_path)

######## DOWNLOADING AND UNZIPPING ALL FILES INTO CORRECT PATH ########

project_dir = "demos/unpaired_ct_lung"
os.chdir(project_dir)

url = "https://zenodo.org/record/3835682/files/training.zip"

# if wget is installed remove following line from comments and comment
# out the fname = 'training.zip' line
# fname = wget.download(url)
fname = "training.zip"

# if training.zip is already downloaded in the correct directory then
# comment out the following line
# os.system("wget " + url)


get_file(os.path.join(os.getcwd(), fname), url)

print("The file ", fname, " has successfully been downloaded!")

data_folder_name = "dataset"
path_to_data_folder = os.path.join(main_path, project_dir, data_folder_name)
if os.path.exists(path_to_data_folder):
    shutil.rmtree(path_to_data_folder)
os.mkdir(path_to_data_folder)

with zipfile.ZipFile(fname, "r") as zip_ref:
    zip_ref.extractall(data_folder_name)

print("Files unzipped!")

os.remove(fname)
os.chdir(main_path)

######## MOVING FILES INTO TRAIN DIRECTORY ########

path_to_train = os.path.join(main_path, project_dir, data_folder_name, "train")
path_to_test = os.path.join(main_path, project_dir, data_folder_name, "test")
path_to_images_and_labels = os.path.join(
    main_path, project_dir, data_folder_name, "training"
)

labels_fnames = os.listdir(os.path.join(path_to_images_and_labels, "lungMasks"))
images_fnames = os.listdir(os.path.join(path_to_images_and_labels, "scans"))

if os.path.exists(path_to_train) is not True:
    os.mkdir(path_to_train)
    os.mkdir(os.path.join(path_to_train, "fixed_images"))
    os.mkdir(os.path.join(path_to_train, "fixed_labels"))
    os.mkdir(os.path.join(path_to_train, "moving_images"))
    os.mkdir(os.path.join(path_to_train, "moving_labels"))


def move_files_into_correct_path(
    fnames, path_to_images_and_labels, new_path, suffix, sub_folder_name
):
    os.chdir(os.path.join(path_to_images_and_labels, sub_folder_name))
    for file in fnames:
        if "insp" in file:
            source = file
            destination = os.path.join(path_to_train, "fixed_" + suffix)
            shutil.move(source, destination)
        if "exp" in file:
            source = file
            destination = os.path.join(path_to_train, "moving_" + suffix)
            shutil.move(source, destination)


if os.path.exists(path_to_images_and_labels):
    move_files_into_correct_path(
        images_fnames, path_to_images_and_labels, path_to_train, "images", "scans"
    )
    move_files_into_correct_path(
        labels_fnames, path_to_images_and_labels, path_to_train, "labels", "lungMasks"
    )

os.chdir(main_path)

######## MOVING FILES INTO TEST AND VALID DIRECTORY ########

path_to_test = os.path.join(path_to_data_folder, "test")
path_to_valid = os.path.join(path_to_data_folder, "valid")

if os.path.exists(path_to_test) is not True:

    os.mkdir(path_to_test)
    os.mkdir(os.path.join(path_to_test, "fixed_images"))
    os.mkdir(os.path.join(path_to_test, "fixed_labels"))
    os.mkdir(os.path.join(path_to_test, "moving_images"))
    os.mkdir(os.path.join(path_to_test, "moving_labels"))

    ratio_of_test_and_valid_samples = 0.4

    unique_case_names = []
    for file in images_fnames:
        case_name_as_list = file.split("_")[0:2]
        case_name = case_name_as_list[0] + "_" + case_name_as_list[1]
        unique_case_names.append(case_name)
    unique_case_names = np.unique(unique_case_names)

    test_and_valid_cases = random.sample(
        list(unique_case_names),
        int(ratio_of_test_and_valid_samples * len(unique_case_names)),
    )
    test_cases = test_and_valid_cases[
        0 : int(int(ratio_of_test_and_valid_samples * len(unique_case_names) / 2))
    ]
    valid_cases = test_and_valid_cases[
        int(int(ratio_of_test_and_valid_samples * len(unique_case_names) / 2)) + 1 :
    ]

    def move_test_cases_into_correct_path(test_cases, path_to_train, path_to_test):
        folder_names = os.listdir(path_to_train)
        os.chdir(path_to_train)
        for case in test_cases:
            for folder in folder_names:
                file_names = os.listdir(os.path.join(path_to_train, folder))
                for file in file_names:
                    if case in file:
                        os.chdir(os.path.join(path_to_train, folder))
                        source = file
                        destination = os.path.join(path_to_test, folder)
                        shutil.move(source, destination)

    move_test_cases_into_correct_path(test_cases, path_to_train, path_to_test)

    os.mkdir(path_to_valid)
    os.mkdir(os.path.join(path_to_valid, "fixed_images"))
    os.mkdir(os.path.join(path_to_valid, "fixed_labels"))
    os.mkdir(os.path.join(path_to_valid, "moving_images"))
    os.mkdir(os.path.join(path_to_valid, "moving_labels"))

    move_test_cases_into_correct_path(valid_cases, path_to_train, path_to_valid)

######## NAMING FILES SUCH THAT THEIR NAMES MATCH FOR PAIRING ########

# name all files such that names match exactly for training

for folder in os.listdir(path_to_train):
    path_to_folder = os.path.join(path_to_train, folder)
    os.chdir(path_to_folder)
    for file in os.listdir(path_to_folder):
        if "_insp" in file:
            new_name = file.replace("_insp", "")
        elif "_exp" in file:
            new_name = file.replace("_exp", "")
        else:
            continue
        source = file
        destination = new_name
        os.rename(source, destination)

# name all files such that names match exactly for testing

for folder in os.listdir(path_to_test):
    path_to_folder = os.path.join(path_to_test, folder)
    os.chdir(path_to_folder)
    for file in os.listdir(path_to_folder):
        if "_insp" in file:
            new_name = file.replace("_insp", "")
        elif "_exp" in file:
            new_name = file.replace("_exp", "")
        else:
            continue
        source = file
        destination = new_name
        os.rename(source, destination)

# name all files such that names match exactly for validation

for folder in os.listdir(path_to_valid):
    path_to_folder = os.path.join(path_to_valid, folder)
    os.chdir(path_to_folder)
    for file in os.listdir(path_to_folder):
        if "_insp" in file:
            new_name = file.replace("_insp", "")
        elif "_exp" in file:
            new_name = file.replace("_exp", "")
        else:
            continue
        source = file
        destination = new_name
        os.rename(source, destination)

shutil.rmtree(os.path.join(path_to_images_and_labels))
os.chdir(main_path)

######## FOR UNPAIRED WE USE IMAMGES FROM ONE TIMEPOINT ONLY ########

# so now remove fixed_images and fixed_labels
# and rename moving_images to images
# and moving_labels to labels

folders = os.listdir(os.path.join(project_dir, data_folder_name))

for folder in folders:
    shutil.rmtree(os.path.join(project_dir, data_folder_name, folder, "fixed_images"))
    shutil.rmtree(os.path.join(project_dir, data_folder_name, folder, "fixed_labels"))
    os.rename(
        os.path.join(project_dir, data_folder_name, folder, "moving_images"),
        os.path.join(project_dir, data_folder_name, folder, "images"),
    )
    os.rename(
        os.path.join(project_dir, data_folder_name, folder, "moving_labels"),
        os.path.join(project_dir, data_folder_name, folder, "labels"),
    )

print("All files moved and restructured")

os.chdir(main_path)

######## NOW WE RESACLE THE IMAGES TO 255 ########

data_dir = "demos/unpaired_ct_lung/dataset"
folders = os.listdir(data_dir)

for folder in folders:
    subfolders = os.listdir(os.path.join(data_dir, folder))
    print("\n Working on ", folder, ", progress:")
    for subfolder in tqdm(subfolders):
        files = os.listdir(os.path.join(data_dir, folder, subfolder))
        for file in files:
            if file.startswith("case_020"):  # this case did not laod correctly
                os.remove(os.path.join(data_dir, folder, subfolder, file))
            else:
                im_data = np.asarray(
                    nib.load(os.path.join(data_dir, folder, subfolder, file)).dataobj,
                    dtype=np.float32,
                )
                if np.max(im_data) > 255.0:
                    im_data = ((im_data + 285) / (3770 + 285)) * 255.0  # rescale image
                    img = nib.Nifti1Image(im_data, affine=None)
                    nib.save(img, os.path.join(data_dir, folder, subfolder, file))
                    if np.max(img.dataobj) > 255.0:
                        print(
                            "Recheck the following file: ",
                            os.path.join(data_dir, folder, subfolder, file),
                        )
                    nib.save(img, os.path.join(data_dir, folder, subfolder, file))

######## DOWNLOAD MODEL CKPT FROM MODEL ZOO ########

url = "https://github.com/DeepRegNet/deepreg-model-zoo/raw/master/demo/unpaired_ct_lung/20210110.zip"

fname = "pretrained.zip"

os.chdir(os.path.join(main_path, project_dir))

get_file(os.path.join(os.getcwd(), fname), url)

with zipfile.ZipFile(fname, "r") as zip_ref:
    zip_ref.extractall(os.path.join(data_folder_name, "pretrained"))

# remove pretrained.zip
os.remove(fname)
print("Pretrained model downloaded")
