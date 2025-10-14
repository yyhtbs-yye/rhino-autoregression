import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv1x1(nn.Conv2d):
    """Thin wrapper for clarity."""
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__(in_ch, out_ch, kernel_size=1, bias=bias)

class MLP1x1(nn.Module):
    """Conv-only MLP: 1x1 -> GELU -> Dropout -> 1x1 -> Dropout"""
    def __init__(self, channels, expansion=4, dropout=0.0, bias=True):
        super().__init__()
        hidden = channels * expansion
        self.fc1 = Conv1x1(channels, hidden, bias=bias)
        self.fc2 = Conv1x1(hidden, channels, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MaskedMHSA2d(nn.Module):
    """
    Channels-first causal (autoregressive) multi-head self-attention for images.

    Input / Output: x (B, C, H, W) -> (B, C, H, W)

    Notes
    - Q,K,V are produced by 1x1 convs (fused here).
    - Attention is computed across spatial positions in raster order (H*W),
      using shapes (B, heads, dim, S), never (B, S, C).
    - Optional attn_mask can hide padded spatial tokens:
        * shape (B, H, W) or (B, 1, H, W) or flattened (B, S) with True = mask out.
    """
    def __init__(
        self,
        channels,
        num_heads,
        dropout=0.0,
        bias=True,
        max_seq_len=None,
    ):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # qkv and output projection via 1x1 conv
        self.qkv = Conv1x1(channels, 3 * channels, bias=bias)
        self.proj = Conv1x1(channels, channels, bias=bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Optional cached causal mask for speed (S up to max_seq_len)
        if max_seq_len is not None:
            m = torch.triu(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool), diagonal=1)
            self.register_buffer("causal_mask", m, persistent=False)
        else:
            self.causal_mask = None

    def _causal(self, S, device, dtype=torch.bool):
        if self.causal_mask is not None and self.causal_mask.size(0) >= S:
            m = self.causal_mask[:S, :S]
        else:
            m = torch.triu(torch.ones(S, S, dtype=dtype, device=device), diagonal=1)
        # (1,1,S,S) so it can broadcast over batch and heads
        return m.view(1, 1, S, S)

    @staticmethod
    def _flatten_spatial(x):
        # x: (B, C, H, W) -> (B, C, S) with S=H*W
        return x.flatten(2)

    @staticmethod
    def _maybe_flatten_mask(attn_mask, H, W):
        if attn_mask is None:
            return None
        if attn_mask.dim() == 4:  # (B, 1, H, W) or (B, Cmask, H, W)
            attn_mask = attn_mask[:, 0]  # take first channel
        if attn_mask.dim() == 3:  # (B, H, W)
            attn_mask = attn_mask.flatten(1)  # (B, S)
        # else assume already (B, S)
        return attn_mask

    def forward(self, x, attn_mask=None):
        B, C, H, W = x.shape
        S = H * W

        qkv = self.qkv(x)                       # (B, 3C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)    # each (B, C, H, W)

        # Split heads: (B, C, H, W) -> (B, heads, head_dim, H, W) -> (B, heads, head_dim, S)
        def split_heads(z):
            z = z.view(B, self.num_heads, self.head_dim, H, W)
            return z.flatten(3)  # (B, heads, head_dim, S)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Attention scores: (B, H, S, S) = (B, H, S, D) @ (B, H, D, S)
        attn_scores = torch.matmul(q.transpose(-2, -1), k) * self.scale  # (B, heads, S, S)

        # Causal mask (upper triangular True -> mask)
        causal = self._causal(S, x.device)
        attn_scores = attn_scores.masked_fill(causal, torch.finfo(attn_scores.dtype).min)

        # Optional key padding mask over spatial positions
        kpm = self._maybe_flatten_mask(attn_mask, H, W)
        if kpm is not None:
            kpm = kpm.view(B, 1, 1, S)  # broadcast across heads and query positions
            attn_scores = attn_scores.masked_fill(kpm, torch.finfo(attn_scores.dtype).min)

        # Softmax over key positions + dropout
        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)

        # Context: (B, H, D, S) = (B, H, D, S) @ (B, H, S, S)^T
        ctx = torch.matmul(v, attn.transpose(-2, -1))  # (B, heads, head_dim, S)

        # Merge heads back to channels: (B, C, S) -> (B, C, H, W)
        ctx = ctx.reshape(B, self.channels, S).reshape(B, self.channels, H, W)

        out = self.proj(ctx)
        out = self.proj_drop(out)
        return out
