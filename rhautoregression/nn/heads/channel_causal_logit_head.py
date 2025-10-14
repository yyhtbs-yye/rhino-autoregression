import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from rhautoregression.nn.ops.masked_conv2d import MaskedConv2d

class ChannelCausalLogitHead(nn.Module):
    """
    Channel-causal discrete head WITH internal token embeddings.

    Inputs:
        h: (B, D, H, W)  spatial-causal backbone features (safe at center)
        x: (B, C, H, W)  integer tokens in [0, vocab_size-1]
    Outputs:
        logits: (B, V, C, H, W)
    """
    def __init__(self, in_channels, vocab_size, d_embed, h_channels = 256):

        super().__init__()
        self.in_channels = in_channels
        self.vocab_size = vocab_size
        self.dE = d_embed
        self.hc = h_channels

        # -------- type maps (rgb,rgb,rgb,...) --------

        self.token_emb = nn.Embedding(vocab_size, d_embed)

        # target_embs channels are grouped as in_channels per embed dim: [R0,G0,B0, R1,G1,B1, ...]
        t_in_targets = torch.arange(in_channels).repeat(d_embed)              # length dE*C: 0,1,2,0,1,2,...

        # hidden channels use interleaved types 0,1,2,0,1,2,...
        t_hidden = torch.arange(h_channels) % in_channels

        # fuse input is [hidden_from_targets, trunk_proj]; make backbone type=0 so it's
        # visible to ALL outputs at center under Mask 'B' (since 0 <= out_type).
        t_in_fuse = torch.cat([t_hidden, torch.zeros(h_channels, dtype=torch.long)])

        # final logits channels are (in_channels*vocab_size) laid out as [R,G,B, R,G,B, ...]
        t_out_logits = torch.arange(in_channels).repeat(vocab_size)                    # length C*vocab_size

        # -------- layers --------
        # 1) From target_embs -> hidden (Mask 'A': center uses strict < channel order)
        self.conv_in = MaskedConv2d(
            'A', in_channels * d_embed, h_channels, kernel_size=1, n_types=in_channels, enable_channel_causality=True,
            type_map_in=t_in_targets, type_map_out=t_hidden
        )

        # 3) Fuse hidden-from-targets with backbone (Mask 'B': center uses <= channel order)
        self.conv_fuse = MaskedConv2d(
            'B', 2 * h_channels, h_channels, kernel_size=1, n_types=in_channels, enable_channel_causality=True,
            type_map_in=t_in_fuse, type_map_out=t_hidden
        )

        # 4) Produce logits (still Mask 'B', typed per (channel, class))
        self.conv_out = MaskedConv2d(
            'B', h_channels, in_channels * vocab_size, kernel_size=1, n_types=in_channels, enable_channel_causality=True,
            type_map_out=t_out_logits
        )

    def forward(self, h, x):
        """
        target_embs must be interleaved as [R0,G0,B0, R1,G1,B1, ...] with size B, E*C, H, W
        x must be integer tokens with size (B, C, H, W).
        Returns logits shaped (B, V, C, H, W).
        """
        B, _, H, W = h.shape
        assert x.dim() == 4 and x.shape[1] == self.in_channels, \
            f"x must be (B, C, H, W) with C={self.in_channels}"
        assert x.dtype in (torch.long, torch.int64), "x must be integer (torch.long)."

        # Embed tokens then interleave as [R0,G0,B0, R1,G1,B1, ...] -> (B, dE*C, H, W)
        target_embs = self.token_emb(x)                 # (B, C, H, W, dE)
        target_embs = rearrange(target_embs, 'b c h w e -> b (e c) h w')

        # Optionally stop grad from teacher-forced inputs:
        # target_embs = target_embs.detach()

        hidden_from_tgt = F.relu(self.conv_in(target_embs))   # Mask 'A' 
        fuse = torch.cat([hidden_from_tgt, h], dim=1)

        fuse = F.relu(self.conv_fuse(fuse))                   # Mask 'B'
        logits = self.conv_out(fuse)                          # (B, V*C, H, W), interleaved [R,G,B, R,G,B, ...]

        # reshape to (B, V, C, H, W) to use CE per channel easily
        logits = rearrange(logits, 'b (v c) h w -> b v c h w', c=self.in_channels, v=self.vocab_size)
        return logits

# ---------------------- simple causality test ----------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, C, H, W = 2, 3, 16, 16
    V = 256
    E = 32          # d_embed
    HC = 120        # divisible by C; also used as backbone feature dim D

    model = ChannelCausalLogitHead(in_channels=C, vocab_size=V, d_embed=E, h_channels=HC).to(device)
    model.eval()

    # Inputs:
    #  - backbone features h must be (B, HC, H, W)
    #  - x must be integer tokens (B, C, H, W)
    h = torch.randn(B, HC, H, W, device=device)
    x = torch.randint(0, V, (B, C, H, W), dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(h, x)

    # pick a reference location/channel
    y0, x0, ch = H // 2, W // 2, 1  # center pixel, green channel if RGB
    base = logits[0, :, ch, y0, x0].clone()

    # --- 1) Forbidden info should NOT change the output ---
    h_forbid = h.clone()
    x_forbid = x.clone()

    # (a) spatial future: rows strictly below y0, and columns strictly to the right on row y0
    h_forbid[:, :, y0 + 1:, :] = torch.randn_like(h_forbid[:, :, y0 + 1:, :])
    h_forbid[:, :, y0, x0 + 1:] = torch.randn_like(h_forbid[:, :, y0, x0 + 1:])

    # Replace tokens in forbidden spatial region with random values
    x_forbid[:, :, y0 + 1:, :] = torch.randint(0, V, x_forbid[:, :, y0 + 1:, :].shape, device=device)
    x_forbid[:, :, y0, x0 + 1:] = torch.randint(0, V, x_forbid[:, :, y0, x0 + 1:].shape, device=device)

    # (b) later channels at the center for tokens: for ch=1 (G), later channels are type 2 (B)
    later_ch = list(range(ch + 1, C))
    for t in later_ch:
        x_forbid[:, t, y0, x0] = torch.randint(0, V, x_forbid[:, t, y0, x0].shape, device=device)

    with torch.no_grad():
        logits_forbid = model(h_forbid, x_forbid)
    test_forbid = logits_forbid[0, :, ch, y0, x0]
    diff_forbid = (base - test_forbid).abs().max().item()
    print(f"[FORBIDDEN] max abs diff @ (y0={y0}, x0={x0}, ch={ch}): {diff_forbid:.6g}  (should be ~0)")

    # --- 2) Allowed info SHOULD change the output ---
    h_allow = h.clone()
    x_allow = x.clone()

    # modify some clearly allowed past positions (above or left of center)
    h_allow[:, :, :y0, :x0] += 0.1 * torch.randn_like(h_allow[:, :, :y0, :x0])

    # change tokens at allowed past positions
    x_allow[:, :, :y0, :x0] = torch.randint(0, V, x_allow[:, :, :y0, :x0].shape, device=device)

    # change same channel token at the center (type == ch)
    x_allow[:, ch, y0, x0] = torch.randint(0, V, x_allow[:, ch, y0, x0].shape, device=device)

    # change earlier channel tokens at the center (types < ch)
    for t in range(ch):
        x_allow[:, t, y0, x0] = torch.randint(0, V, x_allow[:, t, y0, x0].shape, device=device)

    with torch.no_grad():
        logits_allow = model(h_allow, x_allow)
    test_allow = logits_allow[0, :, ch, y0, x0]
    diff_allow = (base - test_allow).abs().max().item()
    print(f"[ALLOWED]  max abs diff @ (y0={y0}, x0={x0}, ch={ch}): {diff_allow:.6g}  (should be > 0)")

    # Quick pass/fail criteria
    eps = 1e-6
    ok_forbid = diff_forbid < 5e-7  # tight if integer masks in fp32
    ok_allow = diff_allow > 1e-4

    print(f"CAUSALITY TEST: {'PASS' if (ok_forbid and ok_allow) else 'FAIL'}")
