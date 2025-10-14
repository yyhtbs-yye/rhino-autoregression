import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from rhautoregression.nn.ops.masked_conv2d import MaskedConv2d
from rhautoregression.nn.packs.masked_msa_block import MaskedMHSABlock

class SpatialOnlyARMHSABackbone(nn.Module):
    def __init__(
        self,
        vocab_size,
        in_channels,
        hidden_channels=384,
        n_blocks=8,
        kernel_size=3,
        embed_dim=32,
        num_heads=8,
        resid_dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        assert vocab_size >= 2
        assert in_channels >= 1
        assert hidden_channels >= in_channels, "hidden_channels should be >= in_channels"
        # (Optional but recommended) even split of hidden channels across types
        assert hidden_channels % in_channels == 0, \
            "hidden_channels should be divisible by in_channels for even type cycling."

        self.V = vocab_size
        self.C = in_channels
        self.HC = hidden_channels
        self.E = embed_dim

        # ---- Per-value embedding shared across channels ----
        # Applied independently to each (B,C,H,W) integer.
        self.embedding = nn.Embedding(self.V, self.E)

        # Type map for the first layer inputs: [0..C-1] repeated E times
        type_map_in_first = torch.arange(self.C).repeat(self.E)

        # ---- First layer: Mask A (no self-connection at the center) ----
        self.conv_in = MaskedConv2d(mask_type='A', in_channels=self.C * self.E, out_channels=self.HC, kernel_size=kernel_size,
                                    enable_channel_causality=False, n_types=self.C, type_map_in=type_map_in_first)

        self.act_in = nn.ReLU(inplace=True)
        self.blocks = nn.ModuleList([
            MaskedMHSABlock(channels=self.HC, num_heads=num_heads, attn_dropout=attn_dropout, resid_dropout=resid_dropout)
            for _ in range(n_blocks)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W) Long in [0, V-1]
        returns: logits (B, V, C, H, W)
        """
        assert x.dtype == torch.long, "Expect LongTensor with class indices."
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W) for grayscale images
        B, C, H, W = x.shape
        eh = self.embedding(rearrange(x, 'b c h w -> b (h w) c'))  # B C H W -> B, H*W, C, E
        z = rearrange(eh, 'b (h w) c e -> b (e c) h w', h=H, w=W)  # B, H*W, C, E -> B, E*C, H, W
        h = self.act_in(self.conv_in(z))                            # [RGB]*emb -> R->G->B casual_masking -> [RGB]*hidden
        
        for block in self.blocks:
            h = block(h, None)
        
        return h
