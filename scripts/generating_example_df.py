

import h5py as h5
import pandas as pd
import numpy as np


source_file = pd.read_csv("./data/source_file_txt")
target_file = pd.read_csv("./data/target_file_txt")

source_list = list()
for i in source_file["source "]:
    f = h5.File(i, 'r')
    source_list.append(f["vect"][:])
source_np = np.vstack(source_list)


target_list = list()
for i in target_file["target"]:
    f = h5.File(i, 'r')
    target_list.append(f["vect"][:])

target_np = np.vstack(target_list)

data = np.vstack([source_np,target_np])

sample_name = np.hstack([np.repeat("A1", 40),
np.repeat("A2", 40),
np.repeat("A3", 40),
np.repeat("A4", 40)])

class_name = np.hstack([np.repeat("B1", 80), 
np.repeat("B2", 80)])


data_df = pd.DataFrame(data)

data_df["L1"] = sample_name
data_df["L2"] = class_name

data_df  = data_df.set_index(["L1", "L2"])
data_df.to_csv("data/example_df.csv")