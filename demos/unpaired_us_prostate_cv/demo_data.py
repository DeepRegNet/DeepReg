"""
Download the demo data
"""
import os
import shutil

from tensorflow.keras.utils import get_file

PROJECT_DIR = r"demos/unpaired_us_prostate_cv"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
DATA_REPO = "dataset_trus3d-master"
ZIP_PATH = "master.zip"
ORIGIN = "https://github.com/ucl-candi/dataset_trus3d/archive/master.zip"

get_file(os.path.abspath(ZIP_PATH), ORIGIN, extract=True)

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
shutil.move(DATA_REPO, DATA_PATH)
os.remove(ZIP_PATH)

print("TRUS 3d data downloaded: %s." % os.path.abspath(DATA_PATH))
