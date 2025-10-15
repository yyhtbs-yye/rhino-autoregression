import math
import torch
import torch.nn.functional as F

from rhautoregression.nn.utils.sample_from_discretized_mix_logistic import _split_params

class DMOLNLLoss(torch.nn.Module):
    """Discretized Mixture of Logistics (DMOL) negative log-likelihood loss. """
    def __init__(self, nr_mix, in_channels, reduction="mean"):
        super().__init__()
        self.nr_mix = nr_mix
        self.in_channels = in_channels
        self.reduction = reduction

    def forward(self, x, params):

        return discretized_mix_logistic_loss(
            x, params, self.nr_mix, self.in_channels, self.reduction
        )

# -----------------------------
# DMOL negative log-likelihood
# -----------------------------
def discretized_mix_logistic_loss(
    x,
    params,
    nr_mix,
    in_channels,
    reduction="mean",
    *,
    coupled=False,
):
    """
    Discretized Mixture of Logistics (DMoL) NLL supporting arbitrary channels.

    Args:
      x:      (B, C, H, W) values in [-1, 1]
      params: (B, n_out, H, W) from your DMOL head
      nr_mix: number of mixtures, K
      in_channels: number of channels, C
      reduction: "mean" | "sum" | None
      coupled: if True, apply triangular cross-channel coupling:
               mean[c] <- mean[c] + sum_{j < c} coeff[c,j] * x[j]

    Shapes (after splitting):
      logit_probs: (B, K, H, W)
      means:       (B, K, C, H, W)
      log_scales:  (B, K, C, H, W)
      coeffs:      (B, K, C*(C-1)/2, H, W) if coupled else None
                   stored in row-wise triangular order:
                   (1←0), (2←0), (2←1), (3←0), (3←1), (3←2), ...
    """

    # 8-bit discretization in [-1, 1]
    bin_half = 1.0 / 255.0
    bin_full_log = math.log(2.0 / 255.0)

    # uses the generalized splitter from earlier:
    #   logit_probs: (B,K,H,W)
    #   means: (B,K,C,H,W)
    #   log_scales: (B,K,C,H,W)
    #   coeffs: (B,K,n_coeff,H,W) or None
    logit_probs, means, log_scales, coeffs = _split_params(
        params, nr_mix, in_channels, coupled=coupled
    )

    # Stabilize scales
    log_scales = torch.clamp(log_scales, min=-7.0)
    inv_stdv = torch.exp(-log_scales)

    # (B,1,C,H,W) for broadcasting against (B,K,C,H,W)
    x_ = x.unsqueeze(1)

    # Optionally apply triangular coupling to the channel means
    if coupled:
        # Build adjusted means: mean[c] += sum_{j<c} coeff[c,j] * x[j]
        C = in_channels
        means_c = means.clone()
        offset = 0
        # coeffs shape: (B,K,n_coeff,H,W) with n_coeff = C*(C-1)//2
        for c in range(1, C):
            n_c = c  # number of coeffs targeting channel c (from j=0..c-1)
            co_c = coeffs[:, :, offset:offset + n_c]  # (B,K,n_c,H,W)
            offset += n_c
            # x inputs for j< c : (B,1,n_c,H,W) shares H,W and broadcasts over K
            x_prev = x_[:, :, :n_c]  # (B,1,c,H,W)
            # sum over previous channels j< c
            adj = (co_c * x_prev).sum(dim=2)  # (B,K,H,W)
            means_c[:, :, c] = means[:, :, c] + adj
    else:
        means_c = means

    # Standard DMoL likelihood pieces
    plus_in = (x_ + bin_half - means_c) * inv_stdv
    min_in  = (x_ - bin_half - means_c) * inv_stdv
    cdf_plus = torch.sigmoid(plus_in)
    cdf_min  = torch.sigmoid(min_in)

    mid_in = (x_ - means_c) * inv_stdv
    log_pdf_mid = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    diff = cdf_plus - cdf_min
    log_prob_interval = torch.where(
        diff > 1e-5, torch.log(torch.clamp(diff, min=1e-12)),
        log_pdf_mid + bin_full_log
    )

    log_cdf_plus = F.logsigmoid(plus_in)
    log_one_minus_cdf_min = F.logsigmoid(-min_in)
    x_lt_min = (x_ < -0.999)
    x_gt_max = (x_ > 0.999)

    # (B,K,C,H,W)
    log_probs_chan = torch.where(
        x_lt_min, log_cdf_plus,
        torch.where(x_gt_max, log_one_minus_cdf_min, log_prob_interval)
    )

    # Sum over channels, add mixture logits, then log-sum-exp over mixtures
    log_probs = torch.sum(log_probs_chan, dim=2)                   # (B,K,H,W)
    log_probs = log_probs + F.log_softmax(logit_probs, dim=1)      # (B,K,H,W)
    log_probs = torch.logsumexp(log_probs, dim=1)                  # (B,H,W)

    nll = -log_probs
    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    return nll
