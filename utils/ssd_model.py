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

        self.layer1 = ResidualLayer(1, in_channels=out_ch, out_channels=out_ch)
        self.layer2 = ResidualLayer(1, in_channels=out_ch, out_channels=out_ch*2)
        self.layer3 = ResidualLayer(1, in_channels=out_ch*2, out_channels=out_ch*4)
        self.layer4 = ResidualLayer(1, in_channels=out_ch*4, out_channels=out_ch*8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.maxpool(out)

        out = self.layer2(out)
        source1 = out
        out = self.maxpool(out)

        out = self.layer3(out)
        out = self.maxpool(out)

        source2 = self.layer4(out)

        return source1, source2


class Extras(nn.Module):
    def __init__(self, in_ch):
        super(Extras, self).__init__()
        self.conv1_1 = nn.Conv2d(in_ch, int(in_ch/4), kernel_size=(1))
        self.conv1_2 = nn.Conv2d(int(in_ch/4), int(in_ch/2), kernel_size=(3), stride=3, padding=7)

        self.conv2_1 = nn.Conv2d(int(in_ch/2), int(in_ch/8), kernel_size=(1))
        self.conv2_2 = nn.Conv2d(int(in_ch/8), int(in_ch/4), kernel_size=(3), stride=2, padding=1)

        self.conv3_1 = nn.Conv2d(int(in_ch/4), int(in_ch/8), kernel_size=(1))
        self.conv3_2 = nn.Conv2d(int(in_ch/8), int(in_ch/4), kernel_size=(3))

        self.conv4_1 = nn.Conv2d(int(in_ch/4), int(in_ch/8), kernel_size=(1))
        self.conv4_2 = nn.Conv2d(int(in_ch/8), int(in_ch/4), kernel_size=(3))

    def forward(self, x):
        source3 = self.conv1_2(self.conv1_1(x))
        source4 = self.conv2_2(self.conv2_1(source3))
        source5 = self.conv3_2(self.conv3_1(source4))
        source6 = self.conv4_2(self.conv4_1(source5))

        return source3, source4, source5, source6

class L2Norm(nn.Module):
    def __init__(self, in_ch, scale):
        super(L2Norm, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_ch))
        self.scale = scale
        self.reset_parameter()
        self.eps = 1e-10
    
    def reset_parameter(self):
        init.constant_(self.weight, self.scale)
    
    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)

        weights = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)

        out =weights * x
        return out