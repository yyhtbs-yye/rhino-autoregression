from typing import Optional
import torch
import torch.nn as nn

from rhautoregression.nn.ops.shifted_conv2d import ShiftedConv2d

class GatedResidualBlock(nn.Module):
    """
    Gated ResNet block with shifted convolutions and optional 'aux' input
    (typically the v-stack feature for the h-stack).
    """
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        shift: str,
        dropout: float = 0.0,
        aux_channels: int = 0,
    ):
        super().__init__()
        self.act = nn.ELU(inplace=False)
        self.conv1 = ShiftedConv2d(channels, channels, kernel_size, shift)
        self.dropout = nn.Dropout2d(dropout) if dropout and dropout > 0 else nn.Identity()
        self.pre_gate = ShiftedConv2d(channels, 2 * channels, 1, shift)
        self.aux_to_gate = nn.Conv2d(aux_channels, 2 * channels, kernel_size=1) if aux_channels > 0 else None
        self.out = nn.Conv2d(channels, channels, kernel_size=1)

        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x: torch.Tensor, aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.act(x)
        h = self.conv1(h)
        h = self.dropout(h)

        gate = self.pre_gate(self.act(h))
        if aux is not None and self.aux_to_gate is not None:
            gate = gate + self.aux_to_gate(aux)

        a, b = torch.chunk(gate, chunks=2, dim=1)
        h = torch.tanh(a) * torch.sigmoid(b)
        h = self.out(h)
        return x + h
