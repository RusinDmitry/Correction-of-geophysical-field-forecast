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
        output = self.linear(x.view(*x.shape[:-3], self.channels*210*280)) #[4, 32, 176400] -> [4,32,3]
        o_input = torch.split(x, 3, dim=-3) #[4,32,3,210,280] -> [4,32,3,210,280]
        output = o_input[0].permute(3, 4, 0, 1, 2) + output #[210, 280, 4, 32, 3] +[4,32,3] -> [210,280, 4, 32,3]
        return output.permute(2, 3, 4, 0, 1) # [4,32,3,210,280]



class MyTestModel(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        self.linear1 = torch.nn.Linear(channels*210*280, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 3)

    def forward(self, x):
        '''
        input: S*B*C*H*W
        :param input:
        :return:
        '''
        o_input = torch.split(x, 3, dim=-3)
        x = self.linear1(x.view(*x.shape[:-3], self.channels*210*280)) #[4,32,3*210*280] [4,32,176400] -> [4,32,200]
        x = self.activation(x)
        x = self.linear2(x)
        output = o_input[0].permute(3, 4, 0, 1, 2) + x #[210, 280, 4, 32, 3] + [4, 32, 3]
        return output.permute(2, 3, 4, 0, 1)


import torch.nn as nn


def conv_plus_conv(in_channels: int, out_channels: int):
    """
    Makes UNet block
    :param in_channels: input channels
    :param out_channels: output channels
    :return: UNet block
    """
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.2),
        nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        nn.BatchNorm2d(num_features=out_channels),
        nn.LeakyReLU(0.2),
    )


class UNET(nn.Module):
    def __init__(self, channels=6):
        super().__init__()

        base_channels = channels
        self.channels = channels
        self.down1 = conv_plus_conv(channels*210*280, base_channels)
        self.down2 = conv_plus_conv(base_channels, base_channels * 2)

        self.up1 = conv_plus_conv(base_channels * 2, base_channels)
        self.up2 = conv_plus_conv(base_channels * 4, base_channels)

        self.bottleneck = conv_plus_conv(base_channels * 2, base_channels * 2)

        self.out = nn.Conv2d(in_channels=base_channels, out_channels=channels*210*280, kernel_size=1)

        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # x.shape = (N, N, 3)

        residual1 = self.down1(x.view(*x.shape[:-3], self.channels*210*280))  # x.shape: (N, N, 3) -> (N, N, base_channels)
        x = self.downsample(residual1)  # x.shape: (N, N, base_channels) -> (N // 2, N // 2, base_channels)

        residual2 = self.down2(x)  # x.shape: (N // 2, N // 2, base_channels) -> (N // 2, N // 2, base_channels * 2)
        x = self.downsample(residual2)  # x.shape: (N // 2, N // 2, base_channels * 2) -> (N // 4, N // 4, base_channels * 2)

        # LATENT SPACE DIMENSION DIM = N // 4
        # SOME MANIPULATION MAYBE
        x = self.bottleneck(x)  # x.shape: (N // 4, N // 4, base_channels * 2) -> (N // 4, N // 4, base_channels * 2)
        # SOME MANIPULATION MAYBE
        # LATENT SPACE DIMENSION DIM = N // 4

        x = nn.functional.interpolate(x, scale_factor=2)  # x.shape: (N // 4, N // 4, base_channels * 2) -> (N // 2, N // 2, base_channels * 2)
        x = torch.cat((x, residual2), dim=1)  # x.shape: (N // 2, N // 2, base_channels * 2) -> (N // 2, N // 2, base_channels * 4)
        x = self.up2(x)  # x.shape: (N // 2, N // 2, base_channels * 4) -> (N // 2, N // 2, base_channels)

        x = nn.functional.interpolate(x, scale_factor=2)  # x.shape: (N // 2, N // 2, base_channels) -> (N, N, base_channels)
        x = torch.cat((x, residual1), dim=1)  # x.shape: (N, N, base_channels) -> (N, N, base_channels * 2)
        x = self.up1(x)  # x.shape: (N, N, base_channels * 2) -> (N, N, base_channels)

        x = self.out(x)  # x.shape: (N, N, base_channels) -> (N, N, 3)

        return x