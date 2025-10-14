import torch
import torch.nn as nn
import torch.nn.functional as F

from rhautoregression.nn.ops.masked_conv2d import MaskedConv2d

class ResMaskedConvBlock(nn.Module):
    def __init__(self, in_channels, k=7, enable_channel_causality=True, n_types=3, p=0.0):
        super().__init__()
        self.conv1 = MaskedConv2d('B', in_channels, in_channels, k, n_types=n_types, enable_channel_causality=enable_channel_causality)
        self.act1  = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv2d('B', in_channels, in_channels, k, n_types=n_types, enable_channel_causality=enable_channel_causality)
        self.drop  = nn.Dropout2d(p) if p > 0 else nn.Identity()
    def forward(self, x):
        h = self.act1(self.conv1(x))
        h = self.drop(self.conv2(h))
        return F.relu(x + h)