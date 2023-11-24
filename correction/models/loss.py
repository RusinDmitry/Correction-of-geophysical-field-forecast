from torch import nn
import torch
from correction.config import cfg


class TurbulentMSE(nn.Module):
    def __init__(self, meaner, beta=1, logger=None):
        super().__init__()
        self.mean_mse = nn.MSELoss()
        self.delta_mse = nn.MSELoss()
        self.meaner = meaner
        self.beta = beta
        if logger:
            logger.set_beta(beta)

    def forward(self, orig, corr, target, logger=None):
        mean_corr = self.meaner(corr)

        t = target.view(*target.shape[:-2], target.shape[-1] * target.shape[-2])
        t = t[..., self.meaner.mapping.unique().long()]

        mse1 = self.mean_mse(mean_corr, t)
        delta_corr = corr.permute(-2, -1, *list(range(len(corr.shape)-2)))-corr.mean(dim=(-1, -2))
        delta_orig = orig.permute(-2, -1, *list(range(len(orig.shape)-2)))-orig.mean(dim=(-1, -2))
        mse2 = self.delta_mse(delta_corr, delta_orig)
        if logger:
            logger.accumulate_stat(mse1 + self.beta*mse2, mse1, mse2)
        return mse1 + self.beta*mse2