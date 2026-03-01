"https://arxiv.org/pdf/1505.04597"

from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import ReLU
import torch.nn as nn
import torch

class Block(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Block, self).__init__()
        self.conv1 = Conv2d(inChannels, outChannels, kernel_size=3, stride=1)
        self.relu = ReLU()
        self.conv2 = Conv2d(outChannels, outChannels, kernel_size=3, stride=1)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.conv2(x)

        return x

class ImageEncoder(nn.Module):
    def __init__(self, channels=(1, 64, 128, 256, 512, 1024)):
            super().__init__()
            self.encBlocks = nn.ModuleList([
                Block(channels[i], channels[i+1]) for i in range(len(channels)-1)
            ])
            self.pool = MaxPool2d(2) 
            self.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        #loop over all the blocks
        for block in self.encBlocks:
            x = block(x)
            x = self.pool(x)
        #apply avg max pooling in all the 1024 layers
        fvect = self.global_pool(x).squeeze(-1) #squeeze to change from [B, 1024, 1, 1] to [B, 1024,1]
        
        return fvect, x

