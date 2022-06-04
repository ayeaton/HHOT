import pandas as pd
import numpy as np
import ot
import torch
from geomloss import SamplesLoss
from utils import HHOT, OT_calc
import argparse

def main(dset1_path, dset2_path, 
        level1_dset1_path, level1_dset2_path,
        level2_dset1_path, level2_dset2_path,
        p, blur, debias, use_pot,
        reg, out_path):

    dset1 = pd.read_csv(dset1_path,index_col=[0,1])
    dset2 = pd.read_csv(dset2_path,index_col=[0,1])

    level1_dset1 = pd.read_csv(level1_dset1_path)
    level1_dset2 = pd.read_csv(level1_dset2_path)

    level2_dset1 = pd.read_csv(level2_dset1_path)
    level2_dset2 = pd.read_csv(level2_dset2_path)

    # initialize OT dist mat
    dist_mat0 = np.zeros((level1_dset1.shape[0], 
                        level1_dset2.shape[0]))

    # loop through and populate dist mat
    for idxi, i in enumerate(level1_dset1["l1_d1"]):
        for idxj, j in enumerate(level1_dset2["l1_d2"]):
            current_dset1_data = dset1.loc[i]
            current_dset2_data = dset2.loc[j]
            dist0 = OT_calc(np.array(current_dset1_data), 
                            np.array(current_dset2_data), 
                            p, blur, debias, use_pot, reg)
            dist_mat0[idxi, idxj] = dist0
    
    dist_mat0_df = pd.DataFrame(dist_mat0)
    dist_mat0_df.to_csv(out_path + "/OT_mat.csv")

    # initialize HHOT dist mat
    dist_mat1 = np.zeros((len(np.unique(level2_dset1["l2_d1"])), 
                        len(np.unique(level2_dset2["l2_d2"]))))

    for idxgi, i in enumerate(np.unique(level2_dset1["l2_d1"])):
        for idxgj, j in enumerate(np.unique(level2_dset2["l2_d2"])):
            dset1_group_idx = np.where(level2_dset1["l2_d1"] == i)
            dset2_group_idx = np.where(level2_dset2["l2_d2"] == j)
            tmp = np.take(dist_mat0, list(dset1_group_idx), axis = 0)
            tohhot = np.take(tmp[0], list(dset2_group_idx), axis = 1)
            dist1 = HHOT(tohhot[:,0,:], reg)
            dist_mat1[idxgi, idxgj] = dist1

    dist_mat1_df = pd.DataFrame(dist_mat1)
    dist_mat1_df.to_csv(out_path + "/HHOT_mat.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dset1_path', default = "data/example_df.csv")
    parser.add_argument('--dset2_path', default = "data/example_df.csv")
    parser.add_argument('--level1_dset1_path', default = "data/l1_d1")
    parser.add_argument('--level1_dset2_path', default = "data/l1_d2")
    parser.add_argument('--level2_dset1_path', default = "data/l2_d1")
    parser.add_argument('--level2_dset2_path', default = "data/l2_d2")
    parser.add_argument('--p', default = 2)
    parser.add_argument('--blur', default = 0.5)
    parser.add_argument('--debias', default = True)
    parser.add_argument('--use_pot', default = True)
    parser.add_argument('--reg', default = 0.03)
    parser.add_argument('--out_path', default = "data/")
    args = parser.parse_args()

    main(dset1_path = args.dset1_path,
        dset2_path = args.dset2_path, 
        level1_dset1_path = args.level1_dset1_path,
        level1_dset2_path = args.level1_dset2_path,
        level2_dset1_path = args.level2_dset1_path,
        level2_dset2_path = args.level2_dset2_path,
        p = args.p,
        blur = args.blur,
        debias = args.debias,
        use_pot = args.use_pot,
        reg = args.reg,
        out_path = args.out_path)

