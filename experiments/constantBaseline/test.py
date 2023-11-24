import sys

import numpy as np

sys.path.insert(0, '../../')
# sys.path.insert(0, '/app/Precipitation-Nowcasting-master')

import torch
from torch.utils.data import DataLoader
from correction.config import cfg
from correction.models.loss import TurbulentMSE
import os

from correction.models.changeToERA5 import MeanToERA5
from correction.data.train_test_split import split_train_val_test
from correction.data.scalers import StandardScaler
from correction.data.my_dataloader import WRFNPDataset
from correction.test import test_model
from correction.models.constantBias import ConstantBias

batch_size = 64
run_id = 1
best_epoch = 0
use_spatiotemporal_encoding = False
wrf_folder = '/home/wrf_data/'
era_folder = '/home/era_data/'

print('Splitting train val test...')
_, _, test_files = split_train_val_test(wrf_folder, era_folder, 0.7, 0.1, 0.2)
print('Split completed!')
folder_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[-1]
# logger = WRFLogger(cfg.GLOBAL.MODEL_SAVE_DIR, folder_name)

era_scaler = StandardScaler()
wrf_scaler = StandardScaler()
era_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_means')),
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'era_stds')))
wrf_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means')),
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds')))

test_dataset = WRFNPDataset(test_files[0][:32], test_files[1][:32])
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

meaner = MeanToERA5(os.path.join(cfg.GLOBAL.BASE_DIR, 'wrferaMapping.npy'))
criterion = TurbulentMSE(meaner, beta=0.5, logger=None).to(cfg.GLOBAL.DEVICE)

model = ConstantBias(6 if use_spatiotemporal_encoding is True else 3).to(cfg.GLOBAL.DEVICE)
model.load_state_dict(torch.load(os.path.join(cfg.GLOBAL.BASE_DIR, f'logs/ConstantBaseline/misc_{run_id}/models/model_{best_epoch}.pth')))
pytorch_total_params = sum(p.numel() for p in model.parameters())
# loss_values = [0. for _ in range(len(losses))]

save_dir = f'logs/ConstantBaseline/misc_{run_id}/'

if __name__ == '__main__':
    metric = test_model(model, criterion, wrf_scaler, era_scaler, test_dataloader, save_dir)
    print(metric)
