import torch
import torch.nn as nn
from typing import Tuple

from rhautoregression.nn.utils.sample_from_discretized_mix_logistic import sample
from rhautoregression.nn.backbones.pixelcnnpp_backbone import PixelCNNPPBackbone

class PixelCNNPlusPlus(nn.Module):
    """
    PixelCNN++ with shifted convolutions (no masks), 
    dual stacks (down & downright), 
    and DMOL output head.

    It is input-output continuos space.

    """
    def __init__(
        self,
        in_channels,
        hidden_channels=120,
        n_blocks=6,
        kernel_size=3,
        dropout=0.0,
        nr_mix=10,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.HC = hidden_channels
        self.kernel_size = kernel_size
        self.p_drop = float(dropout)

        # Mixture components used by the DMOL head (change if desired)
        self.nr_mix = nr_mix

        self.backbone = PixelCNNPPBackbone(in_channels, hidden_channels=hidden_channels,
                                           n_blocks=n_blocks, kernel_size=kernel_size, dropout=dropout)

        # Output head: sum of skips -> DMOL params
        out_channels = get_n_out_channels_for_dmol(in_channels, self.nr_mix)

        self.head = nn.Sequential(nn.ELU(inplace=False),
                                  nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
                                  nn.ELU(inplace=False),
                                  nn.Conv2d(hidden_channels, out_channels, kernel_size=1))

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
          x: (B, C, H, W) in [-1, 1]
        Returns:
          DMOL params: (B, n_out, H, W)
        """
        assert x.dtype == torch.float32, "Expect float input value."

        h = self.backbone(x)

        return self.head(h)

    @torch.no_grad()
    def sample(self, x0):
        return sample(model=self, x0=x0)

def get_n_out_channels_for_dmol(in_channels: int, nr_mix: int, *, coupled: bool = False) -> int:
    """
    Number of output channels for a DMOL head with arbitrary input channels.
    - Always predict: K logits, C*K means, C*K log_scales.
    - If `coupled=True`, also predict triangular channel-coupling coeffs:
      n_coeff = C*(C-1)/2 per mixture (PixelCNN++ style).
    """
    if in_channels < 1:
        raise ValueError("in_channels must be >= 1")
    n_coeff = (in_channels * (in_channels - 1)) // 2 if coupled else 0
    return nr_mix * (1 + 2 * in_channels + n_coeff)

# --- Optional smoke test ---
if __name__ == "__main__":

    from typing import List, Tuple, Dict

    @torch.inference_mode(False)
    def assert_pixelcnnpp_causality(
        model: torch.nn.Module,
        in_channels,
        H=9,
        W=9,
        positions=0,         # 0 = test all pixels, else test this many random pixels
        seed=0,
        atol=1e-12,        # absolute tolerance for tiny numerical noise
        rtol=1e-6,         # relative tolerance scaled by max grad magnitude per check
        verbose: bool = True,
    ) -> Dict:
        """
        Gradient-based causality test for PixelCNN++-style models with shifted convolutions.

        For many pixel locations (i, j), we:
        1) run a forward pass,
        2) sum the output logits/params at (i, j),
        3) backprop to the input,
        4) verify that gradients on any forbidden inputs (self + future pixels in raster order)
            are identically zero (up to small tolerances).

        Args:
            model:       Your PixelCNNPlusPlus instance (should be on the intended device).
            in_channels: Number of input channels the model expects.
            H, W:        Spatial size to test with (kept modest to keep the test fast).
            positions:   0 to test all H*W positions, or a positive integer to test a random subset.
            seed:        RNG seed when sampling a subset of positions.
            atol, rtol:  Absolute/relative tolerances for treating grads as zero.
            verbose:     Print a short human-readable summary.

        Returns:
            A dict with fields:
                passed (bool): whether all checks passed
                violations (list): each item describes a leaking (i, j) with offending positions
                tested_positions (int): number of (i, j) tested
        """
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()  # turn off dropout etc.

        # Freeze parameter grads (we only need input grads)
        reqs = []
        for p in model.parameters():
            reqs.append(p.requires_grad)
            p.requires_grad_(False)

        # Random input
        B, C = 1, in_channels
        x = torch.randn(B, C, H, W, device=device, requires_grad=True)

        # Choose which (i, j) to test
        all_positions: List[Tuple[int, int]] = [(i, j) for i in range(H) for j in range(W)]
        if positions and positions < len(all_positions):
            g = torch.Generator(device=device).manual_seed(seed)
            perm = torch.randperm(len(all_positions), generator=g, device=device).tolist()
            test_set = [all_positions[k] for k in perm[:positions]]
        else:
            test_set = all_positions

        violations = []
        # Keep graph and reuse it; we’ll backward repeatedly from different output pixels.
        # That means calling backward with retain_graph=True and manually zeroing x.grad.
        y = model(x)  # shape: (1, n_out, H, W)

        for (i, j) in test_set:
            # Scalar objective: sum all output channels at this (i, j)
            scalar = y[0, :, i, j].sum()

            # Zero input grad from previous iteration
            if x.grad is not None:
                x.grad.zero_()

            scalar.backward(retain_graph=True)

            # Reduce grad magnitude across channels -> (H, W)
            grad_map = x.grad.detach().abs().sum(dim=1)[0]  # (H, W)
            maxg = float(grad_map.max().item())
            eps = atol + rtol * maxg

            # Forbidden region: self and all pixels "after" (i, j) in raster order
            #   allowed: (r < i) or (r == i and c < j)
            #   forbidden: everything else (including (i, j) itself)
            allowed_mask = torch.zeros(H, W, dtype=torch.bool, device=device)
            if j > 0:
                allowed_mask[i, :j] = True
            if i > 0:
                allowed_mask[:i, :] = True
            forbidden_mask = ~allowed_mask

            # Locations with suspicious (non-zero) grad in forbidden region
            bad = (grad_map > eps) & forbidden_mask
            if bad.any():
                # Collect all offending coordinates
                locs = torch.nonzero(bad, as_tuple=False).tolist()
                violations.append({
                    "pixel": (i, j),
                    "max_grad": maxg,
                    "tolerance": eps,
                    "offenders": [(int(r), int(c), float(grad_map[r, c].item())) for r, c in locs],
                })

        # Restore param requires_grad
        for p, r in zip(model.parameters(), reqs):
            p.requires_grad_(r)
        if was_training:
            model.train()

        passed = (len(violations) == 0)
        if verbose:
            if passed:
                print(f"[Causality ✓] No leakage detected across {len(test_set)} positions "
                    f"on a {H}x{W} grid (in_channels={in_channels}).")
            else:
                first = violations[0]
                pi, pj = first["pixel"]
                print(f"[Causality ✗] Leakage detected at output pixel (i={pi}, j={pj}). "
                    f"First few offending inputs (r, c, |grad|): {first['offenders'][:5]}")

        return {
            "passed": passed,
            "violations": violations,
            "tested_positions": len(test_set),
        }

    # Example usage with your class:
    model = PixelCNNPlusPlus(in_channels=3, hidden_channels=64, n_blocks=4, kernel_size=3, dropout=0.0, nr_mix=5).to("cuda")
    res = assert_pixelcnnpp_causality(model, in_channels=3, H=9, W=9, positions=40, seed=0)
    print(res["passed"], f"violations: {len(res['violations'])}")
    # pass
