import h5py
from utils import HHOT
import argparse

def main(h5py_file, reg):
    h5py_cont = h5py.File(h5py_file, 'r')
    h5py_arr = h5py_cont["OT_dist"][:]
    d = HHOT(h5py_arr, reg = reg)
    print(d)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5py_file', default= "data/ot_par/concat.h5")
    parser.add_argument('--reg', default = 10)
    args = parser.parse_args()

    main(h5py_file = args.h5py_file,
    reg = args.reg)
