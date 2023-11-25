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



class MyTestModel3(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        self.activation = torch.nn.ReLU()
        # Encoder
        self.conv1_1 = torch.nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=3)
        self.bn1 = torch.nn.BatchNorm3d(32)
        self.pool1 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.dropout1 = nn.Dropout()

        self.conv1_2 = torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3)
        self.bn2 = torch.nn.BatchNorm3d(64)
        self.pool2 = torch.nn.MaxPool3d(kernel_size=(2, 2, 2))
        self.dropout2 = nn.Dropout()

        # Decoder
        self.conv2_1 = torch.nn.Conv3d(in_channels=64, out_channels=32, kernel_size=3, padding=3)
        self.bn3 = torch.nn.BatchNorm3d(32)
        self.upsampling1 = torch.nn.Upsample(scale_factor=2)
        self.dropout3 = nn.Dropout()

        self.conv2_2 = torch.nn.Conv3d(in_channels=32, out_channels=3, kernel_size=3, padding=3)

        self.dropout4 = nn.Dropout()

    def forward(self, x):
        '''
        input: S*B*C*H*W
        :param input:
        :return:
        '''
        o_input = torch.split(x, 3, dim=-3)
        x = x.permute(1, 2, 0, 3, 4)

        # Encoder
        x = self.conv1_1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        x = self.conv1_2(x)
        x = self.activation(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Decoder
        x = self.conv2_1(x)
        x = self.activation(x)
        x = self.bn3(x)
        x = self.upsampling1(x)
        x = self.dropout3(x)

        x = self.conv2_2(x)
        x = self.activation(x)
        x = torch.nn.Upsample(size = (o_input[0].shape[0], 210, 280))(x)
        x = self.dropout4(x)

        x = x.permute(2, 0, 1, 3, 4) + o_input[0]

        return x


class MyTestModel(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        self.activation = torch.nn.ReLU()
        # Encoder
        self.conv1_1 = torch.nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=3)
        self.bn1 = torch.nn.BatchNorm3d(32)
        self.dropout1 = nn.Dropout()

        # Decoder
        self.conv2_1 = torch.nn.Conv3d(in_channels=32, out_channels=3, kernel_size=3, padding=3)
        self.dropout3 = nn.Dropout()
        self.upsample1 = torch.nn.Upsample(size = (4, 210, 280))

    def forward(self, x):
        '''
        input: S*B*C*H*W
        :param input:
        :return:
        '''
        o_input = torch.split(x, 3, dim=-3)
        x = x.permute(1, 2, 0, 3, 4)

        # Encoder
        x = self.conv1_1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        # Decoder
        x = self.conv2_1(x)
        x = self.activation(x)
        x = self.upsample1(x)
        x = self.dropout3(x)

        x = x.permute(2, 0, 1, 3, 4)
        return x + o_input[0]

class MyTestModel4(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        self.activation = torch.nn.ReLU()
        # Encoder
        self.conv1_1 = torch.nn.Conv3d(in_channels=3, out_channels=32, kernel_size=3, padding=3)
        self.bn1 = torch.nn.BatchNorm3d(32)
        self.dropout1 = nn.Dropout()

        self.f1 = torch.nn.Linear(64)
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout()


        # Decoder
        self.conv2_1 = torch.nn.Conv3d(in_channels=32, out_channels=3, kernel_size=3, padding=3)
        self.dropout3 = nn.Dropout()
        self.upsample1 = torch.nn.Upsample(size = (4, 210, 280))

    def forward(self, x):
        '''
        input: S*B*C*H*W
        :param input:
        :return:
        '''
        o_input = torch.split(x, 3, dim=-3)
        x = x.permute(1, 2, 0, 3, 4)

        # Encoder
        x = self.conv1_1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.dropout1(x)

        # Decoder
        x = self.conv2_1(x)
        x = self.activation(x)
        x = self.upsample1(x)
        x = self.dropout2(x)

        x = x.permute(2, 0, 1, 3, 4)
        return x + o_input[0]

class MyTestModel2(nn.Module):
    def __init__(self, channels=6):
        super().__init__()
        self.channels = channels
        self.linear1 = torch.nn.Linear(channels*210*280, 2000)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(2000, 1000)
        self.activation = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(1000, 500)
        self.activation = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(500, 200)
        self.activation = torch.nn.ReLU()
        self.linear5 = torch.nn.Linear(200, 3)

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
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = self.activation(x)
        x = self.linear5(x)
        output = o_input[0].permute(3, 4, 0, 1, 2) + x #[210, 280, 4, 32, 3] + [4, 32, 3]
        return output.permute(2, 3, 4, 0, 1)


import torch.nn as nn

from torchvision.transforms.functional import center_crop

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

        base_channels = 200
        self.channels = channels
        self.down1 = conv_plus_conv(channels*210*280, 200)
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