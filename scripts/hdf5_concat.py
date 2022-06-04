import os
import glob
import h5py
from utils import get_hdf5_size, get_keys_type, create_hdf5_dtset_vect
import argparse

def main(input_dir):

    glob_str = input_dir + "/*.h5"
    size, vect_size = get_hdf5_size(glob_str)
    keys_dtype = get_keys_type(glob_str)

    string_dt = h5py.special_dtype(vlen=bytes)
    out_file = os.path.join(input_dir, "concat.h5")

    with h5py.File(out_file ,mode='w') as h5fw:
        # create datasets
        for key in keys_dtype:
            create_hdf5_dtset_vect(h5fw, key, size, vect_size)
        h5fw.create_dataset("source_file", shape=(size , vect_size), dtype= string_dt)
        row = 0
        for h5name in glob.glob(glob_str):
            h5fr = h5py.File(h5name,'r')
            td_key = [key for key in keys_dtype if key[2] > 1]
            dslen = h5fr[td_key[0][0]].shape[0]
            for key in keys_dtype:
                h5fw[key[0]][row:row+dslen,:] = h5fr[key[0]][:]
            h5fw["source_file"][row:row+dslen,:] = h5name
            row += dslen

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default= "data/ot_par")
    args = parser.parse_args()

    main(input_dir = args.input_dir)
