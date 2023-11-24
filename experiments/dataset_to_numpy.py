import numpy as np
from torch.utils.data import Dataset
import netCDF4
import wrf


class LoadDataset(Dataset):
    def __init__(self, wrf_files, era_files,
                 wrf_variables=None, era_variables=None, pattern='*', seq_len=4):
        super().__init__()
        print(len(wrf_files), 'wrf_files')
        print(len(era_files), 'era_files')
        wrf_files.sort()
        era_files.sort()
        self.wrf_files = wrf_files
        self.era_files = era_files
        if era_variables is None:
            era_variables = ['u10', 'v10', 't2m']
        self.era_variables = era_variables
        if wrf_variables is None:
            wrf_variables = ['uvmet10', 'T2']
        self.wrf_variables = wrf_variables

    def load_file_vars(self, filename, variables):
        npy = []
        with netCDF4.Dataset(filename, 'r') as ncf:
            for i, variable in enumerate(variables):
                var = wrf.getvar(ncf, variable, wrf.ALL_TIMES, meta=False)
                if len(var.shape) == 3:
                    var = np.expand_dims(var, 0)
                npy.append(var)
        npy = np.concatenate(npy, 0)
        return np.transpose(npy, (1, 0, 2, 3))

    def __len__(self):
        return len(self.wrf_files)

    def get_data_by_id(self, i, file_attr, var_attr):
        return self.load_file_vars(getattr(self, file_attr)[i], getattr(self, var_attr))

    def __getitem__(self, i):
        data = self.get_data_by_id(i, 'wrf_files', 'wrf_variables')
        target = self.get_data_by_id(i, 'era_files', 'era_variables')
        return data, target


def find_files(directory, pattern):
    import os, fnmatch
    flist = []
    for root, dirs, files in os.walk(directory):
        for basename in files:
            if fnmatch.fnmatch(basename, pattern):
                filename = os.path.join(root, basename)
                filename = filename.replace('\\', '/')
                flist.append(filename)
    return flist


if __name__ == "__main__":
    import sys
    import os
    from tqdm import tqdm
    sys.path.insert(0, '/home/')
    from correction.data.train_test_split import split_train_val_test

    wrf_folder = '/home/wrf_data/'
    era_folder = '/home/era_data/'

    train_files, val_files, test_files = split_train_val_test(wrf_folder, era_folder, 0.8, 0., 0.2)

    train_dataset = LoadDataset(train_files[0], train_files[1])
    for i in tqdm(range(len(train_dataset))):
        wrf_data, era = train_dataset.__getitem__(i)
        np.save(os.path.join('/home/wrf_numpy/train/', train_dataset.wrf_files[i].split('/')[-1]), wrf_data.data)
        np.save(os.path.join('/home/era_numpy/train/', train_dataset.era_files[i].split('/')[-1]), era.data)

    test_dataset = LoadDataset(test_files[0], test_files[1])
    for i in tqdm(range(len(test_dataset))):
        wrf_data, era = test_dataset.__getitem__(i)
        np.save(os.path.join('/home/wrf_numpy/test/', test_dataset.wrf_files[i].split('/')[-1]), wrf_data.data)
        np.save(os.path.join('/home/era_numpy/test/', test_dataset.era_files[i].split('/')[-1]), era.data)

