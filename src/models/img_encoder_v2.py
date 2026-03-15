from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
import torch.nn as nn
import torch


class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Block, self).__init__()
        self.conv1 = Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


class ImageEncoder(nn.Module):
    def __init__(self, channels=(1, 64, 128)):
        super().__init__()

        self.encBlocks = nn.ModuleList([
            Block(channels[i], channels[i + 1]) for i in range(len(channels) - 1)
        ])

        self.pool = MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feature_maps = []

        for block in self.encBlocks:
            x = block(x)
            feature_maps.append(x)   # save feature map before pooling
            x = self.pool(x)

        fvect = self.global_pool(x).flatten(1)   # [B, C]

        return fvect, feature_maps[-1]