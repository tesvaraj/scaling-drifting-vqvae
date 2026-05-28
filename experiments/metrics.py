"""Metrics and diagnostics for VQ-VAE experiments.

Provides:
    - PSNR / SSIM / LPIPS (per-image perceptual metrics)
    - rFID (reconstruction Frechet Inception Distance)
    - Codebook diagnostics: utilization, perplexity, entropy, usage histogram
    - Encoder/codebook geometry: norm stats, code spread, code-to-code distances

LPIPS and FID are imported lazily because they pull in heavy deps (torchvision
models, scipy). If they are unavailable the metric returns None and the
caller logs that gracefully.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F


# ----- pixel-level metrics -----

def psnr(recon: torch.Tensor, x: torch.Tensor, peak: float = 2.0) -> float:
    """PSNR in dB. Inputs are in [-1, 1] by default so peak = 2."""
    mse = F.mse_loss(recon, x).clamp(min = 1e-12)
    return (10.0 * torch.log10(peak ** 2 / mse)).item()


def _gaussian_kernel(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device = device, dtype = dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def ssim(recon: torch.Tensor, x: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5, peak: float = 2.0) -> float:
    """Mean SSIM across batch and channels. Inputs in [-1, 1] so peak = 2.

    Implementation follows the standard Wang et al. (2004) recipe with a
    Gaussian window. No external dependency.
    """
    device, dtype = recon.device, recon.dtype
    window_1d = _gaussian_kernel(window_size, sigma, device, dtype)
    window_2d = (window_1d[:, None] * window_1d[None, :]).unsqueeze(0).unsqueeze(0)
    channels = recon.shape[1]
    window = window_2d.expand(channels, 1, window_size, window_size).contiguous()

    pad = window_size // 2
    mu_x = F.conv2d(x, window, padding = pad, groups = channels)
    mu_r = F.conv2d(recon, window, padding = pad, groups = channels)
    mu_x2 = mu_x.pow(2)
    mu_r2 = mu_r.pow(2)
    mu_xr = mu_x * mu_r

    sigma_x2 = F.conv2d(x * x, window, padding = pad, groups = channels) - mu_x2
    sigma_r2 = F.conv2d(recon * recon, window, padding = pad, groups = channels) - mu_r2
    sigma_xr = F.conv2d(x * recon, window, padding = pad, groups = channels) - mu_xr

    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2

    ssim_map = ((2 * mu_xr + c1) * (2 * sigma_xr + c2)) / \
               ((mu_x2 + mu_r2 + c1) * (sigma_x2 + sigma_r2 + c2))
    return ssim_map.mean().item()


# ----- LPIPS (lazy import) -----

_LPIPS_MODEL = None
_LPIPS_FAILED = False


def lpips(recon: torch.Tensor, x: torch.Tensor, net: str = 'vgg') -> Optional[float]:
    """Perceptual distance via LPIPS. Requires the ``lpips`` package.

    Inputs in [-1, 1], 3 channels (RGB). Returns None if package missing.
    """
    global _LPIPS_MODEL, _LPIPS_FAILED
    if _LPIPS_FAILED:
        return None
    if _LPIPS_MODEL is None:
        try:
            import lpips as _lpips_pkg  # noqa: F401
            _LPIPS_MODEL = _lpips_pkg.LPIPS(net = net, verbose = False).to(recon.device).eval()
            for p in _LPIPS_MODEL.parameters():
                p.requires_grad_(False)
        except Exception:
            _LPIPS_FAILED = True
            return None

    if recon.shape[1] == 1:
        # LPIPS needs 3 channels; replicate grayscale
        recon3 = recon.expand(-1, 3, -1, -1)
        x3 = x.expand(-1, 3, -1, -1)
    else:
        recon3, x3 = recon, x
    with torch.no_grad():
        d = _LPIPS_MODEL(recon3, x3)
    return d.mean().item()


# ----- FID (lazy, uses torchmetrics if available) -----

_FID_METRIC = None
_FID_FAILED = False


def reset_fid():
    global _FID_METRIC
    if _FID_METRIC is not None:
        _FID_METRIC.reset()


def fid_update(real: torch.Tensor, fake: torch.Tensor) -> None:
    """Accumulate one batch into the running FID.

    Inputs should be in [-1, 1]; they are remapped to uint8 [0, 255] internally.
    Silently no-ops if torchmetrics isn't installed.
    """
    global _FID_METRIC, _FID_FAILED
    if _FID_FAILED:
        return
    if _FID_METRIC is None:
        try:
            from torchmetrics.image.fid import FrechetInceptionDistance
            _FID_METRIC = FrechetInceptionDistance(feature = 2048, normalize = False).to(real.device)
        except Exception:
            _FID_FAILED = True
            return

    def to_uint8(t):
        t3 = t if t.shape[1] == 3 else t.expand(-1, 3, -1, -1)
        return ((t3.clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)

    _FID_METRIC.update(to_uint8(real), real = True)
    _FID_METRIC.update(to_uint8(fake), real = False)


def fid_compute() -> Optional[float]:
    if _FID_METRIC is None:
        return None
    try:
        return _FID_METRIC.compute().item()
    except Exception:
        return None


# ----- codebook diagnostics -----

def code_usage_counts(indices: torch.Tensor, codebook_size: int) -> torch.Tensor:
    flat = indices.reshape(-1).detach().cpu()
    return torch.bincount(flat, minlength = codebook_size).float()


def perplexity_from_counts(counts: torch.Tensor) -> float:
    total = counts.sum().clamp(min = 1)
    probs = counts / total
    nz = probs > 0
    entropy = -(probs[nz] * probs[nz].log()).sum()
    return entropy.exp().item()


def entropy_from_counts(counts: torch.Tensor) -> float:
    """Shannon entropy in nats. Max = log(K) for a uniform code distribution."""
    total = counts.sum().clamp(min = 1)
    probs = counts / total
    nz = probs > 0
    return -(probs[nz] * probs[nz].log()).sum().item()


def utilization_from_counts(counts: torch.Tensor) -> float:
    return (counts > 0).float().mean().item()


def gini_from_counts(counts: torch.Tensor) -> float:
    """Gini coefficient over code usage. 0 = perfectly uniform, 1 = one code used.

    Useful supplement to perplexity: catches the case where a few codes
    dominate even though many are alive.
    """
    sorted_counts, _ = counts.sort()
    n = sorted_counts.numel()
    if n == 0 or sorted_counts.sum() <= 0:
        return 0.0
    cum = torch.cumsum(sorted_counts, dim = 0)
    return (1.0 - 2.0 * (cum.sum().item() / (cum[-1].item() * n)) + 1.0 / n)


# ----- geometry -----

@torch.no_grad()
def codebook_geometry(codebook: torch.Tensor) -> dict:
    """Summary stats on code geometry.

    Returns mean/std/min/max of:
        - codebook vector norms
        - pairwise distances between codes
    """
    norms = codebook.norm(dim = -1)
    K = codebook.shape[0]
    # pairwise distances via cdist; for very large K we subsample
    if K > 2048:
        idx = torch.randperm(K, device = codebook.device)[:2048]
        sub = codebook[idx]
    else:
        sub = codebook
    d = torch.cdist(sub, sub)
    upper = d[torch.triu(torch.ones_like(d), diagonal = 1) > 0]
    return {
        'norm_mean': norms.mean().item(),
        'norm_std': norms.std().item(),
        'pair_dist_mean': upper.mean().item() if upper.numel() else 0.0,
        'pair_dist_std': upper.std().item() if upper.numel() > 1 else 0.0,
        'pair_dist_min': upper.min().item() if upper.numel() else 0.0,
    }


@torch.no_grad()
def hidden_geometry(hidden: torch.Tensor) -> dict:
    """Stats on a flat (N, d) batch of encoder outputs."""
    norms = hidden.norm(dim = -1)
    return {
        'hidden_norm_mean': norms.mean().item(),
        'hidden_norm_std': norms.std().item(),
    }


# ----- gradient norms -----

def grad_norm(parameters) -> float:
    total = 0.0
    for p in parameters:
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def grad_norms_per_module(module_dict: dict) -> dict:
    return {name: grad_norm(m.parameters()) for name, m in module_dict.items()}


# ----- aggregator -----

@dataclass
class MetricAccumulator:
    """Accumulate per-batch metrics across a validation pass."""
    psnr_vals: list = field(default_factory = list)
    ssim_vals: list = field(default_factory = list)
    lpips_vals: list = field(default_factory = list)
    rec_losses: list = field(default_factory = list)
    counts: Optional[torch.Tensor] = None

    def add(self, recon, x, codebook_size, indices, want_ssim = True, want_lpips = False,
            want_fid = False):
        self.rec_losses.append(F.mse_loss(recon, x).item())
        self.psnr_vals.append(psnr(recon, x))
        if want_ssim:
            self.ssim_vals.append(ssim(recon, x))
        if want_lpips:
            v = lpips(recon, x)
            if v is not None:
                self.lpips_vals.append(v)
        if want_fid:
            fid_update(x, recon)
        c = code_usage_counts(indices, codebook_size)
        self.counts = c if self.counts is None else self.counts + c

    def summarize(self) -> dict:
        def avg(xs):
            return sum(xs) / len(xs) if xs else float('nan')
        out = {
            'rec_loss': avg(self.rec_losses),
            'psnr': avg(self.psnr_vals),
            'ssim': avg(self.ssim_vals) if self.ssim_vals else float('nan'),
            'lpips': avg(self.lpips_vals) if self.lpips_vals else float('nan'),
        }
        if self.counts is not None:
            out.update({
                'utilization': utilization_from_counts(self.counts),
                'perplexity': perplexity_from_counts(self.counts),
                'entropy_nats': entropy_from_counts(self.counts),
                'gini': gini_from_counts(self.counts),
                'active_codes': int((self.counts > 0).sum().item()),
            })
        return out
