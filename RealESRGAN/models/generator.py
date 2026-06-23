from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from .attention import ChannelAttention
from .blocks import RRDB


class RRDBNetCA(nn.Module):
    """RRDBNet generator with optional channel attention after trunk and upsample stages."""

    def __init__(
        self,
        num_in_ch: int = 3,
        num_out_ch: int = 3,
        scale: int = 4,
        num_feat: int = 64,
        num_block: int = 23,
        num_grow_ch: int = 32,
        ca_reduction: int = 16,
        use_ca_after_trunk: bool = True,
        use_ca_after_up1: bool = True,
        use_ca_after_up2: bool = True,
    ) -> None:
        """Build generator backbone, trunk, upsampling stack, and output head."""

        super().__init__()
        self.scale = scale

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)])
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.ca_trunk = ChannelAttention(num_feat, reduction=ca_reduction) if use_ca_after_trunk else nn.Identity()
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.ca_up1 = ChannelAttention(num_feat, reduction=ca_reduction) if use_ca_after_up1 else nn.Identity()
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.ca_up2 = ChannelAttention(num_feat, reduction=ca_reduction) if use_ca_after_up2 else nn.Identity()
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upscale LR input by 4x and return SR output tensor."""

        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat = self.ca_trunk(feat)

        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = self.lrelu(self.conv_up1(feat))
        feat = self.ca_up1(feat)

        feat = F.interpolate(feat, scale_factor=2, mode="nearest")
        feat = self.lrelu(self.conv_up2(feat))
        feat = self.ca_up2(feat)

        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


def load_pretrained_generator_ca(model: nn.Module, weight_path: str, device: str = "cpu") -> nn.Module:
    """Load pretrained generator weights with tolerant key matching for CA variants."""

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weights not found: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

    if "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
        print("Loaded key: params_ema")
    elif "params" in checkpoint:
        state_dict = checkpoint["params"]
        print("Loaded key: params")
    else:
        state_dict = checkpoint
        print("Loaded raw state_dict")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    ca_missing = [k for k in missing if ("ca_" in k or ".ca." in k)]
    other_missing = [k for k in missing if not ("ca_" in k or ".ca." in k)]
    print(f"\nPretrained keys loaded  : {len(state_dict) - len(unexpected)}")
    print(f"New CA keys (init only) : {len(ca_missing)}")
    if other_missing:
        print(f"WARNING - unexpected missing keys (non-CA): {other_missing}")
    if unexpected:
        print(f"WARNING - unexpected extra keys in ckpt   : {unexpected}")
    print("Pretrained weights loaded successfully into CA generator.")
    return model
