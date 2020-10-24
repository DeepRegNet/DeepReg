import os
import random
import shutil
import zipfile

import nibabel as nib
import numpy as np
from tensorflow.keras.utils import get_file

project_dir = "demos/paired_ct_lung"
data_folder_name = "dataset"


def download_data(main_path):
    # DOWNLOADING AND UNZIPPING ALL FILES INTO CORRECT PATH
    os.chdir(project_dir)

    url = "https://zenodo.org/record/3835682/files/training.zip"
    fname = "training.zip"
    get_file(os.path.join(os.getcwd(), fname), url)

    print("The file ", fname, " has successfully been downloaded!")

    if not os.path.exists(os.path.join(main_path, project_dir, data_folder_name)):
        os.mkdir(os.path.join(main_path, project_dir, data_folder_name))

    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall(data_folder_name)

    print("Files unzipped!")

    os.remove(fname)
    os.chdir(main_path)


def moveTestCasesIntoCorrectPath(test_cases, path_to_train, path_to_test):
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


def moveFilesIntoCorrectPath(
    path_to_train, fnames, path_to_images_and_labels, suffix, sub_folder_name
):
    os.chdir(os.path.join(path_to_images_and_labels, sub_folder_name))
    for file in fnames:
        if "insp" in file:
            source = file
            destination = os.path.join(path_to_train, "fixed_" + suffix)
            shutil.move(source, destination)
        elif "exp" in file:
            source = file
            destination = os.path.join(path_to_train, "moving_" + suffix)
            shutil.move(source, destination)


def move_data(main_path):
    # MOVING FILES INTO TRAIN DIRECTORY
    path_to_data_folder = os.path.join(main_path, project_dir, data_folder_name)
    path_to_train = os.path.join(main_path, project_dir, data_folder_name, "train")
    # path_to_test = os.path.join(main_path, project_dir, data_folder_name, "test") # TODO Not used
    path_to_images_and_labels = os.path.join(
        main_path, project_dir, data_folder_name, "training"
    )

    labels_fnames = os.listdir(os.path.join(path_to_images_and_labels, "lungMasks"))
    images_fnames = os.listdir(os.path.join(path_to_images_and_labels, "scans"))

    if not os.path.exists(path_to_train):
        os.mkdir(path_to_train)
        os.mkdir(os.path.join(path_to_train, "fixed_images"))
        os.mkdir(os.path.join(path_to_train, "fixed_labels"))
        os.mkdir(os.path.join(path_to_train, "moving_images"))
        os.mkdir(os.path.join(path_to_train, "moving_labels"))

    if os.path.exists(path_to_images_and_labels):
        moveFilesIntoCorrectPath(
            path_to_train, images_fnames, path_to_images_and_labels, "images", "scans"
        )
        moveFilesIntoCorrectPath(
            path_to_train,
            labels_fnames,
            path_to_images_and_labels,
            "labels",
            "lungMasks",
        )

    os.chdir(main_path)

    # MOVING FILES INTO TEST AND VALID DIRECTORY

    path_to_test = os.path.join(path_to_data_folder, "test")
    path_to_valid = os.path.join(path_to_data_folder, "valid")

    if not os.path.exists(path_to_test):

        os.mkdir(path_to_test)
        os.mkdir(os.path.join(path_to_test, "fixed_images"))
        os.mkdir(os.path.join(path_to_test, "fixed_labels"))
        os.mkdir(os.path.join(path_to_test, "moving_images"))
        os.mkdir(os.path.join(path_to_test, "moving_labels"))

        ratio_of_test_and_valid_samples = 0.2

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

        moveTestCasesIntoCorrectPath(test_cases, path_to_train, path_to_test)

        os.mkdir(path_to_valid)
        os.mkdir(os.path.join(path_to_valid, "fixed_images"))
        os.mkdir(os.path.join(path_to_valid, "fixed_labels"))
        os.mkdir(os.path.join(path_to_valid, "moving_images"))
        os.mkdir(os.path.join(path_to_valid, "moving_labels"))

        moveTestCasesIntoCorrectPath(valid_cases, path_to_train, path_to_valid)

        return path_to_train, path_to_valid, path_to_test, path_to_images_and_labels


def rename_data(
    main_path, path_to_train, path_to_valid, path_to_test, path_to_images_and_labels
):
    # NAMING FILES SUCH THAT THEIR NAMES MATCH FOR PAIRING
    def uniform_file_name(fname):
        if "_insp" in fname:
            fname = fname.replace("_insp", "")
        if "_exp" in file:
            fname = fname.replace("_exp", "")
        return fname

    # name all files such that names match exactly for training

    for folder in os.listdir(path_to_train):
        path_to_folder = os.path.join(path_to_train, folder)
        os.chdir(path_to_folder)
        for file in os.listdir(path_to_folder):
            new_name = uniform_file_name(file)
            source = file
            destination = new_name
            os.rename(source, destination)

    # name all files such that names match exactly for testing

    for folder in os.listdir(path_to_test):
        path_to_folder = os.path.join(path_to_test, folder)
        os.chdir(path_to_folder)
        for file in os.listdir(path_to_folder):
            new_name = uniform_file_name(file)
            source = file
            destination = new_name
            os.rename(source, destination)

    # name all files such that names match exactly for validation

    for folder in os.listdir(path_to_valid):
        path_to_folder = os.path.join(path_to_valid, folder)
        os.chdir(path_to_folder)
        for file in os.listdir(path_to_folder):
            new_name = uniform_file_name(file)
            source = file
            destination = new_name
            os.rename(source, destination)

    print("All files moved and restructured")

    shutil.rmtree(os.path.join(path_to_images_and_labels))
    os.chdir(main_path)

    # NOW WE NEED TO RESCALE EACH IMAGE

    data_dir = f"demos/paired_ct_lung/{data_folder_name}"
    folders = os.listdir(data_dir)

    file_paths = []
    for folder in folders:
        subfolders = os.listdir(os.path.join(data_dir, folder))
        print("\n Working on ", folder, ", progress:")
        for subfolder in subfolders:
            files = os.listdir(os.path.join(data_dir, folder, subfolder))
            file_paths += [(f, subfolder, folder) for f in files]

    for file, subfolder, folder in file_paths:
        if file.startswith("case_020"):  # this case did not load correctly
            os.remove(os.path.join(data_dir, folder, subfolder, file))
            continue
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


def preprocess_data(main_path):
    path_to_train, path_to_valid, path_to_test, path_to_images_and_labels = move_data(
        main_path
    )
    rename_data(
        main_path, path_to_train, path_to_valid, path_to_test, path_to_images_and_labels
    )


def download_pretrained_model(main_path):
    # DOWNLOAD MODEL CKPT FROM MODEL ZOO

    url = "https://github.com/DeepRegNet/deepreg-model-zoo/raw/master/paired_ct_lung_demo_logs.zip"

    fname = "pretrained.zip"

    os.chdir(os.path.join(main_path, project_dir))

    get_file(os.path.join(os.getcwd(), fname), url)

    with zipfile.ZipFile(fname, "r") as zip_ref:
        zip_ref.extractall(os.path.join(data_folder_name, "pretrained"))

    print("Pretrained model downloaded")


def main():
    main_path = os.getcwd()
    os.chdir(main_path)

    download_data(main_path)
    preprocess_data(main_path)
    download_pretrained_model(main_path)


if __name__ == "__main__":
    main()
