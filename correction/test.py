import sys

sys.path.insert(0, '../')
import torch
import os
from correction.config import cfg
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def test_model(model, criterion, wrf_scaler, era_scaler, dataloader, logs_dir, logger=None):
    test_losses = 0
    i = 0
    overall_metric = 0
    with torch.no_grad():
        model.eval()
        for test_data, test_label in (pbar := tqdm(dataloader)):
            test_data = torch.swapaxes(test_data.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)
            test_label = torch.swapaxes(test_label.type(torch.float).to(cfg.GLOBAL.DEVICE), 0, 1)

            test_data = wrf_scaler.channel_transform(test_data, 2)
            test_label = era_scaler.channel_transform(test_label, 2)

            output = model(test_data)

            output = wrf_scaler.channel_inverse_transform(output, 2)
            test_data = wrf_scaler.channel_inverse_transform(test_data, 2)
            test_label = era_scaler.channel_inverse_transform(test_label, 2)

            sample_metric = calculate_metric(test_data, output, test_label, criterion)
            print(sample_metric)
            overall_metric += sample_metric
            pbar.set_description(f'{sample_metric}')
        overall_metric = overall_metric / len(dataloader)

    return overall_metric


def calculate_metric(wrf_orig, wrf_corr, era, criterion):
    loss_orig = criterion(wrf_orig, wrf_orig, era)
    loss_corr = criterion(wrf_orig, wrf_corr, era)
    print(loss_orig, loss_corr)
    metric = max((loss_orig - loss_corr) / loss_orig, 0)
    return metric
