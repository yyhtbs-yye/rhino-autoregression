import torch
import torch.nn.functional as F

@torch.no_grad()
def sample(model, x0: torch.Tensor, temperature: float = 1.0, greedy: bool = False, device=None) -> torch.Tensor:
    """
    Autoregressive sampling in raster order, channel order 0..C-1.

    Args:
        x0 : (B, C, H, W) Long. Use >=0 to keep a value fixed, <0 to fill.
        temperature : float > 0
        greedy : if True, take argmax; else sample from softmax

    Returns:
        (B, C, H, W) Long in [0, V-1]
    """
    model.eval()
    x = x0.to(device)
    if x.dtype != torch.long:
        x = x.long()

    B, C, H, W = x.shape
    V = model.V

    fixed = (x >= 0)
    x = torch.where(fixed, x, torch.zeros_like(x))

    for y in range(H):
        for xcol in range(W):
            for c in range(C):
                logits = model(x)                # (B, V, C, H, W)
                logits_c = logits[:, :, c, y, xcol]     # (B, V)

                if temperature != 1.0:
                    logits_c = logits_c / float(max(1e-8, temperature))

                if greedy:
                    next_val = logits_c.argmax(dim=-1)        # (B,)
                else:
                    probs = F.softmax(logits_c, dim=-1)
                    probs = torch.nan_to_num(probs, nan=0.0)
                    next_val = torch.multinomial(probs, num_samples=1).squeeze(1)

                x[:, c, y, xcol] = next_val

    return x

@torch.no_grad()
def sample_backbone1_headN(model, x0: torch.Tensor, temperature: float = 1.0, greedy: bool = False, device=None) -> torch.Tensor:
    """
    Autoregressive sampling in raster order, channel order 0..C-1.

    Args:
        x0 : (B, C, H, W) Long. Use >=0 to keep a value fixed, <0 to fill.
        temperature : float > 0
        greedy : if True, take argmax; else sample from softmax

    Returns:
        (B, C, H, W) Long in [0, V-1]
    """
    model.eval()
    x = x0.to(device)
    if x.dtype != torch.long:
        x = x.long()

    B, C, H, W = x.shape
    V = model.V

    fixed = (x >= 0)
    x = torch.where(fixed, x, torch.zeros_like(x))

    for y in range(H):
        for xcol in range(W):
            hiddens = model.backbone(x)                
            for c in range(C):
                
                logits = model.head(hiddens, x)

                logits_c = logits[:, :, c, y, xcol]     # (B, V)

                if temperature != 1.0:
                    logits_c = logits_c / float(max(1e-8, temperature))

                if greedy:
                    next_val = logits_c.argmax(dim=-1)        # (B,)
                else:
                    probs = F.softmax(logits_c, dim=-1)
                    probs = torch.nan_to_num(probs, nan=0.0)
                    next_val = torch.multinomial(probs, num_samples=1).squeeze(1)

                x[:, c, y, xcol] = next_val

    return x
