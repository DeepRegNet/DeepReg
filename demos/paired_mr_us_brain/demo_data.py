import os
import shutil

import nibabel as nib
import numpy as np

main_path = os.getcwd()

project_dir = os.path.join(main_path, r"demos/paired_mr_us_brain")
os.chdir(project_dir)


os.system("git clone https://github.com/ucl-candi/dataset_respect.git")

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
