import pickle as pkl

import h5py as h5
import numpy as np
from tqdm import tqdm

with open("./data/exp1-key-random.pkl", "rb") as f:
    data = pkl.load(f)

lst = data["train"] + data["test"] + data["holdout"]
collection = []
for i in lst:
    collection.append(i[0])
    collection.append(i[1])
collection = list(set(collection))

t2_arr = np.zeros([128, 128, 102])
label_arr = np.zeros([128, 128, 102])

with h5.File(f"./data/ImageWithSeg-0.7-0.7-0.7-64-64-51-SimpleNorm-RmOut.h5", "w") as f:
    for key in tqdm(collection):
        dset = f.create_dataset(name=key, data=[t2_arr, label_arr])


with open("./data/holdout-landmarks-key-random.pkl", "rb") as f:
    data = pkl.load(f)

lst = data["holdout"]
collection = []
for i in lst:
    collection.append(i[0])
    collection.append(i[1])
collection = list(set(collection))
with h5.File(f"./data/holdout-landmarks.h5", "w") as f:
    for key in tqdm(collection):
        dset = f.create_dataset(name=key, data=[t2_arr, label_arr])
