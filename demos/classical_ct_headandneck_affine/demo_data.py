"""
Download the demo data
"""
import os

from tensorflow.keras.utils import get_file

current_path = os.getcwd()
PROJECT_DIR = r"demos/classical_ct_headandneck_affine"
os.chdir(PROJECT_DIR)

DATA_PATH = "dataset"
FILE_PATH = os.path.abspath(os.path.join(DATA_PATH, "demo.h5"))

if os.path.exists(DATA_PATH):
    os.remove(FILE_PATH)
else:
    os.mkdir(DATA_PATH)

ORIGIN = "https://github.com/YipengHu/example-data/blob/master/hnct/demo.h5"

get_file(FILE_PATH, ORIGIN)
print("CT head-and-neck data downloaded: %s." % FILE_PATH)

os.chdir(current_path)
