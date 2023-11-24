from torch import nn
import torch


class ConstantBias(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        self.linear = nn.Linear(channels*210*280, 3)

    def forward(self, x):
        '''
        input: S*B*C*H*W
        :param input:
        :return:
        '''
        output = self.linear(x.view(*x.shape[:-3], self.channels*210*280))
        o_input = torch.split(x, 3, dim=-3)
        output = o_input[0].permute(3, 4, 0, 1, 2) + output
        return output.permute(2, 3, 4, 0, 1)
