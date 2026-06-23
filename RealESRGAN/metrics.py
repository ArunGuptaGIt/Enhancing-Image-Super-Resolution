from __future__ import annotations

import importlib

import torch
import torch.nn.functional as F
from torch import Tensor


def build_lpips_model(device: torch.device):
    """Load LPIPS model (Alex backbone) for perceptual validation metric."""

    import lpips

    lpips_fn = lpips.LPIPS(net="alex").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad = False
    return lpips_fn


def calculate_psnr(sr: Tensor, hr: Tensor, max_val: float = 1.0) -> float:
    """Compute PSNR between SR and HR tensors in [0, 1] range."""

    mse = F.mse_loss(sr, hr)
    if mse == 0:
        return float("inf")
    return (20 * torch.log10(torch.tensor(max_val, device=sr.device) / torch.sqrt(mse))).item()


def calculate_ssim(sr: Tensor, hr: Tensor) -> float:
    """Compute SSIM using skimage for a single image pair."""

    metrics_mod = importlib.import_module("skimage.metrics")
    ssim = getattr(metrics_mod, "structural_similarity")

    sr_np = sr.squeeze().permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
    hr_np = hr.squeeze().permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)
    return float(ssim(sr_np, hr_np, data_range=1.0, channel_axis=2))


@torch.no_grad()
def calculate_lpips(sr: Tensor, hr: Tensor, lpips_fn) -> float:
    """Compute LPIPS score after remapping images from [0, 1] to [-1, 1]."""

    sr_scaled = sr * 2.0 - 1.0
    hr_scaled = hr * 2.0 - 1.0
    return float(lpips_fn(sr_scaled, hr_scaled).item())
