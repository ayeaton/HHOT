import pandas as pd
import numpy as np
import os
import h5py
import ot
import torch
from geomloss import SamplesLoss
import argparse
from utils import HHOT, OT_calc


def main(source_file_txt, target_file_txt, p, 
        blur, debias, use_pot, reg, out_path):

    
    source_files = list(pd.read_csv(source_file_txt).iloc[:,0])
    target_files = list(pd.read_csv(target_file_txt).iloc[:,0])

    for i in range(len(source_files)):

        # create array to save distance and list to save names
        si_target_D = np.zeros((1, len(target_files)))
        target_file_names = list()

        current_source_file = source_files[i]
        out_file = os.path.join(out_path, "OT-" + os.path.basename(current_source_file))

        # read source hdf5
        source_hdf5 = h5py.File(current_source_file, 'r')
        source_hdf5_arr = source_hdf5["vect"][:]
        source_hdf5.close()

        for j in range(len(target_files)):

            current_target_file = target_files[j]

            # read target hdf5
            target_hdf5 = h5py.File(current_target_file, 'r')
            target_hdf5_arr = target_hdf5["vect"][:]
            target_hdf5.close()

            target_file_names.append(os.path.basename(current_target_file)[:]) 

            dist0 = OT_calc(source_hdf5_arr, target_hdf5_arr, p, blur, debias, use_pot, reg)

            si_target_D[0, j] = dist0

        target_file_names_stack = np.stack(target_file_names, axis=0).astype('S16')

        with h5py.File(out_file ,mode='w') as h5fw:
            h5fw.create_dataset("OT_dist", shape=(1, len(target_file_names) ),  dtype='float64')
            h5fw.create_dataset("filenames", shape=(len(target_file_names_stack), ),  dtype='S16')
            h5fw["OT_dist"][:] = si_target_D
            h5fw["filenames"][:] = target_file_names_stack


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file_txt', default = "data/source_file_txt")
    parser.add_argument('--target_file_txt', default = "data/target_file_txt")
    parser.add_argument('--p', default = 2)
    parser.add_argument('--blur', default = 0.5)
    parser.add_argument('--debias', default = True)
    parser.add_argument('--use_pot', default = True)
    parser.add_argument('--reg', default = 10)
    parser.add_argument('--out_path', default = "data/")
    args = parser.parse_args()

main(source_file_txt = args.source_file_txt,
    target_file_txt = args.target_file_txt,
    p = args.p,
    blur = args.blur,
    debias = args.debias,
    use_pot = args.use_pot,
    reg = args.reg,
    out_path = args.out_path)
