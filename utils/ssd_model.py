import numpy as np
import os
import torch
from torchvision import transforms
import torch.utils.data as data
import torch.nn as nn


def conv3x3(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)


def conv1x1(in_channels, out_channels, kernel_size = 1, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)


class BasickBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample = None):
        super(BasickBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity_x = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity_x = self.downsample(x)
        
        out += x
        return self.relu(out)


class ResidualLayer(nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels, block = BasickBlock):
        super(ResidualLayer, self).__init__()
        downsample = None
        if in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels),
                nn.BatchNorm2d(out_channels)
            )
        self.first_block = block(in_channels, out_channels, downsample=downsample)
        self.blocks = nn.ModuleList(block(out_channels, out_channels) for _ in range(num_blocks))
    
    def forward(self, x):
        out = self.first_block(x)
        for block in self.blocks:
            out = block(out)
        
        return out


class Feature_extractor(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Feature_extractor, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, f_out, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)

        self.layer1 = ResidualLayer(2, in_channels=out_ch, out_channels=out_ch)
        self.layer2 = ResidualLayer(2, in_channels=out_ch, out_channels=out_ch*2)
        self.layer3 = ResidualLayer(2, in_channels=out_ch*2, out_channels=out_ch*4)
        self.layer4 = ResidualLayer(2, in_channels=out_ch*4, out_channels=out_ch*8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out


