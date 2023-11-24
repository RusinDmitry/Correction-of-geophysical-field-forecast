import logging
import os
import re

import numpy as np
import torch


class WRFLogger:
    def __init__(self, base_log_dir=None, folder_name=None):
        if base_log_dir is None:
            base_log_dir = '/home/logs'
        if folder_name is None:
            folder_name = 'unknown'
        if not os.path.exists(os.path.join(base_log_dir, folder_name)):
            os.mkdir(os.path.join(base_log_dir, folder_name))
        self.folder_path = os.path.join(base_log_dir, folder_name)

        self.save_dir = os.path.join(self.folder_path, f'misc_{self.get_experiment_number()}')
        self.model_save_dir = os.path.join(self.save_dir, 'models')
        self.log_dir = os.path.join(self.save_dir, 'logs')

        os.makedirs(self.log_dir)
        os.makedirs(self.model_save_dir)

        self.logger = self.create_logger()
        self.logger.info(f"Testing the custom logger for module {__name__}...")

        self.train_loss = []
        self.loss_evolution = []
        self.best_epoch = -1
        self.mse = 0
        self.mse1 = 0
        self.mse2 = 0
        self.iters_counted = 0
        self.beta = None

    def create_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        handler = logging.FileHandler(f"{os.path.join(self.log_dir, __name__)}.log", mode='w')
        formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_experiment_number(self):
        numbers = set()
        for directory in os.listdir(self.folder_path):
            if re.match(r'(misc)_\d+$', directory):
                numbers.add(int(directory.split('_')[-1]))
        return max(numbers) + 1 if len(numbers) else 1

    def set_beta(self, beta):
        self.beta = beta

    def accumulate_stat(self, delta_mse, mse1=None, mse2=None):
        self.mse += float(delta_mse)
        if mse1:
            self.mse1 += float(mse1)
        if mse2:
            self.mse2 += float(mse2)
        self.iters_counted += 1

    def reset_stat(self):
        self.mse = 0
        self.mse1 = 0
        self.mse2 = 0
        self.iters_counted = 0

    def print_stat_readable(self, epoch=None):
        beta = self.beta if self.beta else 'beta'
        if epoch:
            self.logger.info(f"Validation epoch {epoch} successful with val loss:")
        else:
            self.logger.info(f"Validation successful with val loss:")
        if self.mse > 0:
            mse = round(self.mse/self.iters_counted, 5)
            self.logger.info(f"    MSE + deltaMSE: {mse}")
            self.loss_evolution.append(mse)
        if self.mse1 > 0 and self.mse2 > 0:
            self.logger.info(f"    MSE: {round(self.mse1/self.iters_counted, 5)}"
                             f" + {beta} * deltaMSE: {round(self.mse2/self.iters_counted, 5)} ")
        self.reset_stat()

    def save_model(self, model_state_dict, epoch):
        loss = self.loss_evolution
        if len(loss) > 0 and loss.index(min(loss)) == len(loss)-1:
            torch.save(model_state_dict, os.path.join(self.model_save_dir, f'model_{epoch}.pth'))
            old_model_path = os.path.join(self.model_save_dir, f'model_{self.best_epoch}.pth')
            if os.path.exists(old_model_path):
                os.remove(old_model_path)
            self.best_epoch = len(loss)-1
        np.save(os.path.join(self.log_dir, 'val_loss'), np.stack(self.loss_evolution))
        np.save(os.path.join(self.log_dir, 'train_loss'), np.stack(self.train_loss))
        return self.best_epoch
