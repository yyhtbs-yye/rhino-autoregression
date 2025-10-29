import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from rhcompression.nn.multi_scale_vqvae.multi_scale_vqvae import MultiScaleVQVAE 

# ----------------------------
# AdaLN + Transformer blocks (GPT-2-ish)
# ----------------------------

class AdaLN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ln = nn.LayerNorm(dim, elementwise_affine=False)
        # fallback affine when no conditioning is provided
        self.gamma0 = nn.Parameter(torch.zeros(dim))
        self.beta0  = nn.Parameter(torch.zeros(dim))
        # map conditioning to (gamma, beta)
        self.to_gb = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c=None):
        # x: [B, T, D]
        h = self.ln(x)
        if c is None:
            # broadcast to [B, T, D]
            gamma = self.gamma0.view(1, 1, -1).to(h.dtype)
            beta  = self.beta0.view(1, 1, -1).to(h.dtype)
        else:
            # c: [B, D] -> gamma,beta: [B, 1, D]
            gb = self.to_gb(c.to(h.dtype))
            gamma, beta = gb.chunk(2, dim=-1)
            gamma = gamma.unsqueeze(1)
            beta  = beta.unsqueeze(1)

        return h * (1 + gamma) + beta

class MLP(nn.Module):
    def __init__(self, dim, hidden_mult=4, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_mult*dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_mult*dim, dim)
        self.drop= nn.Dropout(drop)
    def forward(self, x):
        return self.drop(self.fc2(self.act(self.fc1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead, drop=0.0):
        super().__init__()
        self.adaln1 = AdaLN(dim)
        self.attn   = nn.MultiheadAttention(dim, nhead, dropout=drop, batch_first=True)
        self.adaln2 = AdaLN(dim)
        self.mlp    = MLP(dim, drop=drop)

    def forward(self, x, c=None, attn_mask=None):
        qkv = self.adaln1(x, c)
        x = x + self.attn(qkv, qkv, qkv, attn_mask=attn_mask, need_weights=False)[0]
        x = x + self.mlp(self.adaln2(x, c))
        return x

# ----------------------------
# 2D learned positional embeddings per scale
# ----------------------------

class Pos2D(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(1, H*W, dim) * 0.02)
        self.H, self.W = H, W
    def forward(self, B):
        return self.emb.expand(B, -1, -1)  # [B, H*W, D]

# ----------------------------
# VAR Token Generator (next-scale prediction)
#   - shared token embedding (same as VQ codebook dim D)
#   - builds a sequence = [prefix tokens (flattened all previous scales), query tokens of target scale]
#   - transformer predicts logits for target scale in parallel
# ----------------------------

class VARTokenGenerator(nn.Module):
    def __init__(self, codebook_size=4096, D=32, depth=24, nhead=16,
                 scales=(1,2,4,8,16), class_cond=False, num_classes=1000, drop=0.0):
        super().__init__()
        self.scales = list(scales)
        self.D = D
        self.class_cond = class_cond

        # token embedding tied with VQ dimension
        self.tok_embed = nn.Embedding(codebook_size, D)

        # optional class token/conditioning
        if class_cond:
            self.class_embed = nn.Embedding(num_classes, D)
        else:
            self.register_parameter('class_embed', None)

        # per-scale 2D position embeddings
        self.pos = nn.ModuleDict()
        for s in self.scales:
            self.pos[str(s)] = Pos2D(D, s, s)

        # a learned "query" type embedding to mark the next-scale queries
        self.query_type = nn.Parameter(torch.zeros(1, 1, D))

        # transformer trunk (GPT-2-ish, decoder-only; no RoPE etc.)
        self.blocks = nn.ModuleList([TransformerBlock(D, nhead, drop=drop) for _ in range(depth)])
        self.out_head = nn.Linear(D, codebook_size)

    def _flatten_tokens(self, idx, s):
        # idx: [B, s, s] -> embeddings + pos: [B, s*s, D]
        e = self.tok_embed(idx.view(idx.shape[0], -1))
        p = self.pos[str(s)](idx.shape[0])
        return e + p

    def _make_queries(self, B, s):
        # query tokens for target scale: just positions + query-type mark
        p = self.pos[str(s)](B)
        return p + self.query_type.expand(B, p.size(1), -1)

    def forward(self, prefix_idxs, target_scale, y=None):
        """
        prefix_idxs: list of token maps [B, s_i, s_i] for scales < target_scale
        target_scale: the integer scale to predict (e.g., 2,4,8,16)
        y: optional class labels [B] for class-conditional AdaLN
        returns:
            logits: [B, target_scale*target_scale, K]
        """
        B = prefix_idxs[0].shape[0]
        seqs = []

        # optional class conditioning vector c
        c = None
        if self.class_cond:
            if y is None: raise ValueError("class labels required for class_cond=True")
            c = self.class_embed(y)  # [B, D]

        # pack prefix (coarse -> fine)
        for idx in prefix_idxs:
            s = idx.shape[-1]
            seqs.append(self._flatten_tokens(idx, s))

        # append queries for the next scale
        S = target_scale
        queries = self._make_queries(B, S)
        x = torch.cat(seqs + [queries], dim=1)  # [B, N_ctx + N_q, D]

        # full attention (parallel prediction inside the scale)
        for blk in self.blocks:
            x = blk(x, c=c, attn_mask=None)

        # read out only the queries (last S*S positions)
        q = x[:, -S*S:, :]
        logits = self.out_head(q)
        return logits

# ----------------------------
# Usage sketch (no training)
# ----------------------------
if __name__ == "__main__":
    # tokenizer
    vqvae = MultiScaleVQVAE(codebook_size=4096, D=32, scales=(1,2,4,8,16))
    x = torch.randn(2, 3, 256, 256)
    idxs, vq_loss = vqvae.encode(x)      # list of [B, s, s], coarse->fine

    # var token generator
    var = VARTokenGenerator(codebook_size=4096, D=32, depth=12, nhead=8,
                            scales=(1,2,4,8,16), class_cond=False)

    # predict the 8x8 map given {1,2,4}
    logits_8 = var(prefix_idxs=idxs[:3], target_scale=8)   # [B, 64, 4096]
    # predict the 16x16 map given {1,2,4,8}
    logits_16 = var(prefix_idxs=idxs[:4], target_scale=16) # [B, 256, 4096]

    # reconstruct from (ground-truth or sampled) tokens
    x_hat = vqvae.decode(idxs)
