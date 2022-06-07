import math
import torch.nn as nn
from .common import ConvBlock, NUM_ROTATIONS


# Basic ConvNet with Pooling layer
def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class ConvNet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, num_classes=64, remove_linear=False):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

        if remove_linear:
            self.fc = None
        else:
            self.fc = nn.Linear(z_dim, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.further = ConvBlock(z_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, NUM_ROTATIONS)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, feature=False):
        x = self.encoder(x)

        further = self.further(x)
        further = self.avgpool(further)
        further = further.view(further.size(0), -1)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.fc is None:
            if feature:
                return x, None
            else:
                return x

        logits = self.fc(x)
        rot_logits = self.fc2(further)

        if feature:
            return x, logits
        
        return logits, rot_logits


def conv4_64(num_classes, remove_linear=False):
    """
        Contructs Conv-4-64
    """
    return ConvNet(3, 64, 64, num_classes, remove_linear=remove_linear)


def conv4_512(num_classes, remove_linear=False):
    """
        Constructs Conv-4-512
    """
    return ConvNet(3, 64, 512, num_classes, remove_linear=remove_linear)