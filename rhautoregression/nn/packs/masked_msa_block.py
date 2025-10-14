import torch
import torch.nn as nn
import torch.nn.functional as F
from rhcore.nn.ops.channel_layer_norm import ChannelLayerNorm

from rhautoregression.nn.ops.masked_mhsa_2d import MLP1x1, MaskedMHSA2d 

class MaskedMHSABlock(nn.Module):
    """
    GPT-style (pre-norm) transformer block for AR image gen, channels-first.
      x -> CLN -> masked MHSA -> +res
         -> CLN -> 1x1 MLP    -> +res

    Args:
      channels:     embedding channels (must be divisible by num_heads)
      num_heads:    attention heads
      attn_dropout: dropout on attention weights and output projection
      resid_dropout: dropout inside the MLP (and can be used as proj drop if desired)
      mlp_expansion: expansion factor for the MLP hidden size
      norm_layer_factory: callable like `lambda c: ChannelLayerNorm(c)` (use your CLN)
      max_seq_len:  optional prebuilt causal mask length (H*W <= max_seq_len for cache)
    """
    def __init__(self, channels, num_heads, attn_dropout=0.0, resid_dropout=0.0, mlp_expansion=4, bias=True, max_seq_len=None,
    ):
        super().__init__()

        self.norm1 = ChannelLayerNorm(channels)
        self.attn = MaskedMHSA2d(
            channels=channels,
            num_heads=num_heads,
            dropout=attn_dropout,
            bias=bias,
            max_seq_len=max_seq_len,
        )
        self.norm2 = ChannelLayerNorm(channels)
        self.mlp = MLP1x1(channels, expansion=mlp_expansion, dropout=resid_dropout, bias=bias)

    def forward(self, x, attn_mask=None):
        x = x + self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------ minimal usage example ------------------
if __name__ == "__main__":

    import random

    def raster_index(h: int, w: int, W: int) -> int:
        return h * W + w

    def mask_positions_earlier_than(h: int, w: int, H: int, W: int, device) -> torch.Tensor:
        """
        Returns (1, 1, H, W) boolean mask where True marks positions strictly earlier
        than (h, w) in row-major raster order.
        """
        s0 = raster_index(h, w, W)
        grid = torch.arange(H * W, device=device).view(H, W)
        m = (grid < s0).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        return m

    @torch.no_grad()
    def check_spatial_causality(
        block: nn.Module,
        H: int,
        W: int,
        C: int,
        B: int = 2,
        num_trials: int = 32,
        delta_scale: float = 1.0,
        atol: float = 1e-6,
        rtol: float = 0.0,
        device: str = "cpu",
    ):
        """
        For each trial:
        - pick a random spatial position p = (hp, wp)
        - make x2 = x1 with only x2[:,:,hp,wp] perturbed
        - assert outputs y1,y2 are identical on ALL positions strictly earlier than p
        """
        block.eval().to(device)

        # Deterministic input
        g = torch.Generator(device=device)
        g.manual_seed(1234)

        x1 = torch.randn(B, C, H, W, generator=g, device=device)
        y1 = block(x1)  # (B,C,H,W)

        # Try multiple random future-pixel edits
        for t in range(num_trials):
            hp = random.randrange(H)
            wp = random.randrange(W)
            # skip the very first pixel occasionally (it's trivially causal)
            if hp == 0 and wp == 0 and num_trials > 1:
                continue

            x2 = x1.clone()
            # Perturb only the chosen spatial site across all channels
            # (ensure a noticeable change while staying finite)
            bump = torch.randn(B, C, 1, 1, generator=g, device=device) * delta_scale
            x2[:, :, hp:hp+1, wp:wp+1] = x2[:, :, hp:hp+1, wp:wp+1] + bump

            y2 = block(x2)

            # Compare on all positions strictly earlier than (hp, wp)
            earlier_mask = mask_positions_earlier_than(hp, wp, H, W, device)  # (1,1,H,W)
            # Broadcast mask to (B,C,H,W)
            earlier_mask_bc = earlier_mask.expand(B, C, H, W)

            diff = (y1 - y2) * earlier_mask_bc
            max_abs = diff.abs().max().item()

            if not torch.allclose((y1 * earlier_mask_bc), (y2 * earlier_mask_bc), rtol=rtol, atol=atol):
                # Find the first offending location for a helpful error
                bad = (diff.abs() > (atol + rtol * y1.abs())).nonzero(as_tuple=False)
                b, c, hh, ww = bad[0].tolist()
                s_bad = raster_index(hh, ww, W)
                s_edit = raster_index(hp, wp, W)
                raise AssertionError(
                    f"Causality violation: editing future pixel (h={hp},w={wp}, s={s_edit}) "
                    f"changed output at earlier site (b={b}, c={c}, h={hh}, w={ww}, s={s_bad}). "
                    f"max|Δ| on earlier sites = {max_abs:.3e}"
                )

        print(f"PASS: spatial causality holds over {num_trials} randomized future edits "
            f"({H}x{W}, C={C}, B={B}); tolerances rtol={rtol}, atol={atol}.")


    # --- Config ---
    B, C, H, W = 2, 128, 16, 16
    heads = 8
    device = "cpu"  # use "cuda" if you want, but CPU avoids nondeterministic kernels

    # Build a deterministic block (dropouts = 0)
    block = MaskedMHSABlock(
        channels=C,
        num_heads=heads,
        attn_dropout=0.0,
        resid_dropout=0.0,
        mlp_expansion=4,
        bias=True,
        max_seq_len=H * W,  # cache mask (optional)
    ).to(device)

    # Quick smoke test: shape
    x = torch.randn(B, C, H, W, device=device)
    y = block(x)
    assert y.shape == x.shape, "Shape mismatch"

    # Run the spatial-only causality test
    check_spatial_causality(
        block=block,
        H=H, W=W, C=C, B=B,
        num_trials=64,     # more trials = stricter test
        delta_scale=3.0,   # larger perturbation to make effects obvious
        atol=1e-6,
        rtol=0.0,
        device=device,
    )
