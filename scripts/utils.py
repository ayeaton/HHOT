from geomloss import SamplesLoss
import ot
import glob
import h5py

def OT_calc(source, target, p = 2, blur = 0.05, debias = True, use_pot = False, reg = 0.03):
    # calculate OT
    if use_pot:
        C = ot.dist(source, target)
        d = ot.sinkhorn2(ot.unif(len(source)),ot.unif(len(target)),C/C.max(),reg)[0]*C.max()
    else: # use geomloss
        source_torch = torch.tensor(source, requires_grad = True).cuda()
        target_torch = torch.tensor(target).cuda()
        loss = SamplesLoss(loss="sinkhorn", p=p, blur=blur, debias=debias, backend = "tensorized")
        d = loss(source_torch, target_torch)
    return d


def HHOT(cost,reg = 0.03):
    d = ot.sinkhorn2(ot.unif(cost.shape[0]),ot.unif(cost.shape[1]),cost/cost.max(),reg)[0]*cost.max()
    return d

def get_hdf5_size(glob_str):
    size_total = 0
    for h5name in glob.glob(glob_str):
        h5fr = h5py.File(h5name,'r')
        arr_data1 = h5fr[list(h5fr.keys())[0]][:]
        size = arr_data1.shape[0]
        size_total = size_total + size
    arr_data1 = h5fr[list(h5fr.keys())[0]][:]
    vect_size = arr_data1.shape[1]
    return size_total, vect_size


def get_keys_type(glob_str):
    h5name = glob.glob(glob_str)[0]
    h5fr = h5py.File(h5name,'r')
    list_key_dtype = [[i, h5fr[i].dtype, len(h5fr[i].shape)] for i in list(h5fr.keys())]
    return list_key_dtype


def create_hdf5_dtset_vect(h5_file_write, keys_dtype, size, vect_size):
        h5_file_write.create_dataset(keys_dtype[0], shape=(size , vect_size), dtype= keys_dtype[1])
