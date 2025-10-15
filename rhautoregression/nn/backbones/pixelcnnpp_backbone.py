import torch
import torch.nn as nn
import torch.nn.functional as F

from rhautoregression.nn.ops.shifted_conv2d import ShiftedConv2d
from rhautoregression.nn.packs.gated_residual_block import GatedResidualBlock
class PixelCNNPPBackbone(nn.Module):

    def __init__(
        self,
        in_channels,           # data channels C
        hidden_channels=384, # prefer divisible by C (for even typing)
        n_blocks=6,
        kernel_size=3,
        dropout=0.0,
    ):
        super().__init__()

        # Initial shifted convs for v- and h-stacks
        self.v_init = ShiftedConv2d(in_channels, hidden_channels, kernel_size, shift="down")
        self.h_init = ShiftedConv2d(in_channels, hidden_channels, kernel_size, shift="downright")

        # Residual stacks, v->h couplings, and skip projections
        self.v_blocks = nn.ModuleList()
        self.h_blocks = nn.ModuleList()
        self.v_to_h = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        for _ in range(n_blocks):
            self.v_blocks.append(
                GatedResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    shift="down",
                    dropout=dropout,
                    aux_channels=0,
                )
            )
            self.v_to_h.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))

            self.h_blocks.append(
                GatedResidualBlock(
                    channels=hidden_channels,
                    kernel_size=kernel_size,
                    shift="downright",
                    dropout=dropout,
                    aux_channels=hidden_channels,  # receives v each block
                )
            )
            self.skip_convs.append(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1))

    def forward(self, x):
        v = self.v_init(x)
        h = self.h_init(x)
        skip = None
        for v_block, to_h, h_block, skip_conv in zip(self.v_blocks, self.v_to_h, self.h_blocks, self.skip_convs):
            v = v_block(v)
            h = h_block(h, aux=to_h(v))
            s = skip_conv(h)
            skip = s if skip is None else skip + s
        return skip
