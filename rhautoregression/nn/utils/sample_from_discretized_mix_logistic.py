import torch
import torch.nn.functional as F

@torch.no_grad()
def sample(model, x0: torch.Tensor, *, coupled: bool = False, device=None) -> torch.Tensor:
    """
    Raster-scan DMoL sampling.
    x0: (B, C, H, W) canvas (e.g., zeros). Returns x in [-1,1].
    """
    x = x0.clone()
    B, C, H, W = x.shape
    K = model.nr_mix

    for y in range(H):
        for xcol in range(W):
            params = model(x)  # (B, n_out, H, W)

            logit_probs, means, log_scales, coeffs = _split_params(
                params, model.nr_mix, model.in_channels, coupled=coupled
            )  # (B,K,H,W), (B,K,C,H,W), (B,K,C,H,W), (B,K,n_coeff,H,W)|None

            # mixture choice per pixel (Gumbel-max)
            u = torch.rand_like(logit_probs)
            g = -torch.log(-torch.log(torch.clamp(u, 1e-6, 1 - 1e-6)))
            sel = torch.argmax(logit_probs + g, dim=1)                           # (B,H,W)
            one_hot = F.one_hot(sel, num_classes=K).permute(0, 3, 1, 2).float()  # (B,K,H,W)

            # pick params for the chosen mixture
            m    = (means      * one_hot.unsqueeze(2)).sum(dim=1)    # (B,C,H,W)
            logs = (log_scales * one_hot.unsqueeze(2)).sum(dim=1)    # (B,C,H,W)
            logs = torch.clamp(logs, min=-7.0)

            if not coupled:
                # independent channels
                u = torch.rand(B, C, device=x.device)
                eps = torch.log(torch.clamp(u, 1e-6, 1-1e-6)) - torch.log(torch.clamp(1-u, 1e-6, 1-1e-6))
                x[:, :, y, xcol] = m[:, :, y, xcol] + torch.exp(logs[:, :, y, xcol]) * eps
            else:
                # triangular coupling: mu_c += sum_{j<c} coeff[c,j] * x_j
                co_sel = (coeffs * one_hot.unsqueeze(2)).sum(dim=1)   # (B, n_coeff, H, W)
                offset = 0
                for c_idx in range(C):
                    mu_c   = m[:, c_idx, y, xcol]
                    log_c  = logs[:, c_idx, y, xcol]
                    if c_idx > 0:
                        n_c   = c_idx
                        co_c  = co_sel[:, offset:offset+n_c, y, xcol]   # (B, c_idx)
                        prev  = x[:, :c_idx, y, xcol]                   # (B, c_idx)
                        mu_c  = mu_c + (co_c * prev).sum(dim=1)
                        offset += n_c
                    u = torch.rand_like(mu_c)
                    eps = torch.log(torch.clamp(u, 1e-6, 1-1e-6)) - torch.log(torch.clamp(1-u, 1e-6, 1-1e-6))
                    x[:, c_idx, y, xcol] = mu_c + torch.exp(log_c) * eps

            x[:, :, y, xcol] = torch.clamp(x[:, :, y, xcol], -1.0, 1.0)

    return x


def _split_params(params: torch.Tensor, nr_mix: int, in_channels: int, *, coupled: bool = False):
    """
    Returns:
      logit_probs: (B, K, H, W)
      means:       (B, K, C, H, W)
      log_scales:  (B, K, C, H, W)
      coeffs:      (B, K, n_coeff, H, W) if coupled else None
                   where n_coeff = C*(C-1)/2 (triangular ordering)
    """
    B, Cout, H, W = params.shape
    if in_channels < 1:
        raise ValueError("in_channels must be >= 1")

    n_coeff = (in_channels * (in_channels - 1)) // 2 if coupled else 0
    expected = nr_mix * (1 + 2 * in_channels + n_coeff)
    if Cout != expected:
        raise ValueError(f"Expected {expected} channels, got {Cout}")

    idx = 0
    # (B, K, H, W)
    logit_probs = params[:, idx:idx + nr_mix]; idx += nr_mix

    # (B, C*K, H, W) -> (B, K, C, H, W)
    m = params[:, idx:idx + in_channels * nr_mix]; idx += in_channels * nr_mix
    ls = params[:, idx:idx + in_channels * nr_mix]; idx += in_channels * nr_mix

    def _to_kc(x):
        B_, C_, H_, W_ = x.shape
        return x.view(B_, in_channels, nr_mix, H_, W_).permute(0, 2, 1, 3, 4).contiguous()

    means = _to_kc(m)
    log_scales = _to_kc(ls)

    coeffs = None
    if coupled:
        # (B, n_coeff*K, H, W) -> (B, K, n_coeff, H, W)
        co = params[:, idx:idx + n_coeff * nr_mix]; idx += n_coeff * nr_mix
        B_, Cc, H_, W_ = co.shape
        coeffs = torch.tanh(
            co.view(B_, n_coeff, nr_mix, H_, W_).permute(0, 2, 1, 3, 4).contiguous()
        )

    return logit_probs, means, log_scales, coeffs

