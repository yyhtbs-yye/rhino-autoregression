import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from rhautoregression.nn.ops.masked_conv2d import MaskedConv2d
from rhautoregression.nn.packs.res_masked_conv_block import ResMaskedConvBlock
from rhautoregression.nn.backbones.spatial_channel_ar_conv_backbone import SpatialChannelARConvBackbone

from rhautoregression.nn.utils.sample_from_logits import sample

class PixelCNN(nn.Module):
    
    def __init__(
        self,
        vocab_size,
        in_channels,            # data channels C
        hidden_channels=384,    # prefer divisible by C (for even typing)
        n_blocks=6,
        kernel_size=7,
        embed_dim=32,
        dropout=0.0,
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
        self.p_drop = float(dropout)

        self.backbone = SpatialChannelARConvBackbone(
            vocab_size=vocab_size,
            in_channels=in_channels,            # data channels C
            hidden_channels=hidden_channels,    # prefer divisible by C (for even typing)
            n_blocks=n_blocks,
            kernel_size=kernel_size,
            embed_dim=embed_dim,
            dropout=dropout,
        )

        # Type map for the output layer: [RGB,RGB,RGB,...] (V times), is it V*C in size?
        type_map_out_head = torch.arange(self.C).repeat(self.V)

        self.head = MaskedConv2d(
            mask_type='B',
            in_channels=self.HC,
            out_channels=self.C * self.V,
            kernel_size=1,
            n_types=self.C,
            type_map_out=type_map_out_head,
        )

        self.reset_parameters()

    # ----- init -----
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, MaskedConv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    # ----- forward -----
    def forward(self, x):
        """
        x: (B, C, H, W) Long in [0, V-1]
        returns: logits (B, V, C, H, W)
        """
        assert x.dtype == torch.long, "Expect LongTensor with class indices."
        if x.dim() == 3:
            x = x.unsqueeze(1)              # (B, 1, H, W) for grayscale images

        B, C, H, W = x.shape

        h = self.backbone(x)

        logits = self.head(h)               # [RGB]*emb -> R->G->B casual_masking -> [RGB]*vocab

        if logits.dim == 4:
            logits = rearrange(logits, "b (v c) h w -> b v c h w", c=C, v=self.V)
        elif logits.dim == 5:
            pass
        else:
            raise RuntimeError(f"Unexpected logits dimension: {logits.dim}")

        return logits

    @torch.no_grad()
    def sample(self, x0, temperature=1.0, greedy=False):
        return sample(model=self, x0=x0, temperature=temperature, greedy=greedy)

# -------------------------
# Tiny demo / causality test
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 16, 16

    model = PixelCNN(
        vocab_size=256,
        in_channels=C,
        hidden_channels=120,   # divisible by 3
        n_blocks=4,
        kernel_size=5,
        embed_dim=32,
        dropout=0.0,
    ).to(device)
    model.eval()

    # forward shape check
    x = torch.randint(0, 256, (B, C, H, W), device=device)
    logits = model(x)
    print("logits shape:", tuple(logits.shape))  # (B, V, C, H, W)

    # simple structural causality check (no dependence on future pixels / later channels)
    y0, x0, ch = H // 2, W // 2, 1
    base = logits[0, :, ch, y0, x0].detach()

    x_masked = x.clone()
    # zero rows strictly below y0, and columns strictly to the right on row y0
    x_masked[:, :, y0 + 1:, :] = 0
    x_masked[:, :, y0, x0 + 1:] = 0
    # zero *later channels* at (y0, x0)
    if ch + 1 < C:
        x_masked[:, ch + 1:, y0, x0] = 0

    logits2 = model(x_masked)
    test = logits2[0, :, ch, y0, x0].detach()

    diff = (base - test).abs().max().item()
    print("causal_diff:", diff)  # should be ~0 (exact 0 in float32 with these masks)

