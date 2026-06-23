from __future__ import annotations

import torch
from torch import nn

from .attention import ChannelAttention


class ResidualDenseBlock(nn.Module):
    """Standard ESRGAN residual dense block."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        """Initialize five-layer dense block with local residual scaling."""

        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming initialization and residual-friendly final layer init."""

        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.conv5.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run dense feature fusion then add scaled residual to input."""

        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class ResidualDenseBlockCA(nn.Module):
    """Residual dense block with an additional channel-attention gate."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, ca_reduction: int = 16) -> None:
        """Initialize dense block and channel-attention head."""

        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.ca = ChannelAttention(num_feat, reduction=ca_reduction)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming initialization and residual-friendly final layer init."""

        for m in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5]:
            nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.constant_(self.conv5.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run dense feature fusion, channel attention, and residual add."""

        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        x5 = self.ca(x5)
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-residual dense block built from three RDB units."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32) -> None:
        """Create stacked RDB blocks with outer residual scaling."""

        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate through three RDB blocks and apply outer residual."""

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBCA(nn.Module):
    """RRDB variant with channel-attention inside and after the stack."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, ca_reduction: int = 16) -> None:
        """Create CA-enabled stacked dense blocks with final attention."""

        super().__init__()
        self.rdb1 = ResidualDenseBlockCA(num_feat, num_grow_ch, ca_reduction)
        self.rdb2 = ResidualDenseBlockCA(num_feat, num_grow_ch, ca_reduction)
        self.rdb3 = ResidualDenseBlockCA(num_feat, num_grow_ch, ca_reduction)
        self.ca = ChannelAttention(num_feat, reduction=ca_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Propagate through CA blocks, refine with CA, then residual add."""

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = self.ca(out)
        return out * 0.2 + x
