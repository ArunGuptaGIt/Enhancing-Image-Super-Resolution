from __future__ import annotations

import torch
from torch import nn


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention layer."""

    def __init__(self, num_feat: int, reduction: int = 8) -> None:
        """Create channel attention projection with stable initialization."""

        super().__init__()
        mid = max(num_feat // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_feat, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, num_feat, bias=True),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.fc[2].weight, 0.0)
        nn.init.constant_(self.fc[2].bias, 3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale each channel using global pooled channel descriptors."""

        b, c, _, _ = x.shape
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
