import torch
import torch.nn as nn 

NUM_ROTATIONS = 4


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, relu=True, pool=True):
        super().__init__()
        self.layers = nn.Sequential()

        self.layers.add_module(
            "Conv",
            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False),
        )
        self.layers.add_module("BatchNorm", nn.BatchNorm2d(out_planes))

        if relu:
            self.layers.add_module("ReLU", nn.ReLU(inplace=True))
        if pool:
            self.layers.add_module("MaxPool", nn.MaxPool2d(kernel_size=2, stride=2, padding=0))

    def forward(self, x):
        out = self.layers(x)
        return out