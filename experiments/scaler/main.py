import sys
import os
import torch
import netCDF4
import numpy as np
import wrf

sys.path.insert(0, '../../')
from correction.data.train_test_split import split_train_val_test, find_files
from correction.data.scalers import StandardScaler
from correction.config import cfg
from tqdm import tqdm

wrf_folder = '/home/wrf_data/'
era_folder = '/home/era_data/'

wrf_files = find_files(wrf_folder, 'wrf*')
era_files = find_files(era_folder, '*')
wrf_tensor = []
era_tensor = []

print(len(wrf_files), len(era_files), 'file_lengths')
# todo не кушать так много оперативки х(
for wrf_file, era_file in tqdm(zip(wrf_files, era_files), total=len(wrf_files)):
    wrf_tensor.append(torch.from_numpy(np.load(wrf_file)))
    era_tensor.append(torch.from_numpy(np.load(era_file)))
print(len(wrf_tensor))
print(len(era_tensor))
wrf_tensor = torch.cat(wrf_tensor)
era_tensor = torch.cat(era_tensor)

era_scaler = StandardScaler()
era_scaler.channel_fit(era_tensor)
torch.save(era_scaler.channel_means, os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_means'))
torch.save(era_scaler.channel_stddevs, os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_stds'))
print(era_scaler.channel_means, era_scaler.channel_stddevs)

wrf_scaler = StandardScaler()
wrf_scaler.channel_fit(wrf_tensor)
torch.save(wrf_scaler.channel_means, os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means'))
torch.save(wrf_scaler.channel_stddevs, os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds'))
print(wrf_scaler.channel_means, wrf_scaler.channel_stddevs)
