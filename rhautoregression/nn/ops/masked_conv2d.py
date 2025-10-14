
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    """
    Exactly the same semantics as the user's MaskedConv2d with type gating:
    - "A": strictly earlier spatial positions; center uses < for channel order
    - "B": strictly earlier spatial positions; center uses <= for channel order
    """
    def __init__(self, mask_type: str,
                 in_channels: int, out_channels: int,
                 kernel_size: int | tuple[int, int] = 3,
                 *, enable_channel_causality=True, n_types: int = 3,
                 type_map_in: torch.Tensor | None = None,
                 type_map_out: torch.Tensor | None = None,
                 bias: bool = True, **kwargs):
        assert mask_type in ("A", "B")
        if isinstance(kernel_size, int):
            kH = kW = kernel_size
        else:
            kH, kW = kernel_size
        if 'padding' not in kwargs:
            kwargs['padding'] = (kH // 2, kW // 2)

        super().__init__(in_channels, out_channels, (kH, kW), bias=bias, **kwargs)
        assert self.groups == 1, "Grouped conv not supported in this mask builder."

        yc, xc = kH // 2, kW // 2

        # --- 1) Spatial half-plane mask (applies to all channel pairs) ---
        mask = torch.ones_like(self.weight, dtype=torch.float32)
        if yc + 1 < kH:
            mask[:, :, yc + 1:, :] = 0.0        # rows strictly below center
        if xc + 1 < kW:
            mask[:, :, yc, xc + 1:] = 0.0       # cols strictly right on center row
        # DO NOT zero the center here; channel gating will decide it.

        # --- 2) Channel gating at the center position only ---
        # Build type labels for input/output channels.

        if enable_channel_causality:        
            if type_map_in is None:
                t_in = torch.arange(in_channels) % n_types              # [Cin]
            else:
                t_in = torch.as_tensor(type_map_in, dtype=torch.long)
                assert t_in.numel() == in_channels

            if type_map_out is None:
                t_out = torch.arange(out_channels) % n_types            # [Cout]
            else:
                t_out = torch.as_tensor(type_map_out, dtype=torch.long)
                assert t_out.numel() == out_channels

            if mask_type == "A":
                center_ok = (t_in[None, :] <  t_out[:, None]).float()   # [Cout, Cin]
            else:  # "B"
                center_ok = (t_in[None, :] <= t_out[:, None]).float()

            mask[:, :, yc, xc] = center_ok

            self.n_types = n_types
            self.register_buffer("type_map_in_buf", t_in)
            self.register_buffer("type_map_out_buf", t_out)
        else:
            # SPATIAL-ONLY causality:
            # Mask A must not see the current pixel at all (any channel)
            # Mask B may see it.
            if mask_type == "A":
                mask[:, :, yc, xc] = 0.0
            # for "B" we leave the center as 1.0

        self.register_buffer("mask", mask)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
