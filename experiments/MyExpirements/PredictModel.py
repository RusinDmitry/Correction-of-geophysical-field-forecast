import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from correction.config import cfg
from correction.data.my_dataloader import WRFNPDataset
from correction.data.scalers import StandardScaler
from correction.models.changeToERA5 import MeanToERA5
from correction.models.constantBias import MyTestModel
from correction.models.loss import TurbulentMSE

Path_list = [['C:/Users/AI Lab/Desktop/Хакатон Москва/train_dataset_train/train/wrf/wrfout_d01_2019-01-02_00%3A00%3A00.npy']]

PATH_model = 'C:/Users/AI Lab/Desktop/Хакатон Москва/RSVbaseline/logs/constantBaseline/misc_32/models/model_4.pth'
batch_size = 1

wrf_scaler = StandardScaler()

wrf_scaler.apply_scaler_channel_params(torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_means')),
                                       torch.load(os.path.join(cfg.GLOBAL.MODEL_SAVE_DIR, 'wrf_stds')))
test_dataset = WRFNPDataset(Path_list[0],[] )
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8)


meaner = MeanToERA5(os.path.join(cfg.GLOBAL.BASE_DIR, 'wrferaMapping.npy'))
criterion = TurbulentMSE(meaner, beta=0.5, logger=None).to(cfg.GLOBAL.DEVICE)

model = MyTestModel(3).to(cfg.GLOBAL.DEVICE)
model.load_state_dict(torch.load(PATH_model))
print(model)
model.eval()

for test_data, _ in (pbar := tqdm(test_dataloader)):
    test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
    test_data = wrf_scaler.channel_transform(test_data, 2)

    output = model(test_data)

    output = wrf_scaler.channel_inverse_transform(output, 2)
    test_data = wrf_scaler.channel_inverse_transform(test_data, 2)
