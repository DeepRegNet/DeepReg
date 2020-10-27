"""
Download the demo data
"""
import os
import shutil

from tensorflow.keras.utils import get_file

MAIN_PATH = os.getcwd()
PROJECT_DIR = "demos/classical_ct_headneck_affine"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
FILE_PATH = os.path.abspath(os.path.join(DATA_PATH, "demo.h5"))
ORIGIN = "https://github.com/YipengHu/example-data/raw/master/hnct/demo.h5"

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.mkdir(DATA_PATH)

get_file(FILE_PATH, ORIGIN)
print("CT head-and-neck data downloaded: %s." % FILE_PATH)

os.chdir(MAIN_PATH)
