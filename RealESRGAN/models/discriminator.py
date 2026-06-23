from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils import spectral_norm

from ..config import Config


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator with spectral normalization."""

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, n_layers: int = 4) -> None:
        """Build convolutional PatchGAN critic producing a patch-wise realism map."""

        super().__init__()
        kw = 4
        padw = 1

        layers: list[nn.Module] = [
            nn.Conv2d(num_in_ch, num_feat, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            layers += [
                spectral_norm(
                    nn.Conv2d(
                        num_feat * nf_mult_prev,
                        num_feat * nf_mult,
                        kernel_size=kw,
                        stride=2,
                        padding=padw,
                        bias=False,
                    )
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            spectral_norm(
                nn.Conv2d(num_feat * nf_mult_prev, num_feat * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(num_feat * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, bias=False)),
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Return patch-level discriminator logits."""

        return self.model(x)


class UNetDiscriminatorSN(nn.Module):
    """U-Net style discriminator with spectral normalization and optional skip adds."""

    def __init__(self, num_in_ch: int = 3, num_feat: int = 64, skip_connection: bool = True) -> None:
        """Initialize encoder-decoder discriminator with spectral normalized convolutions."""

        super().__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Run discriminator forward pass and output realism logits map."""

        x0 = self.lrelu(self.conv0(x))
        x1 = self.lrelu(self.conv1(x0))
        x2 = self.lrelu(self.conv2(x1))
        x3 = self.lrelu(self.conv3(x2))

        x3 = F.interpolate(x3, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x4 = self.lrelu(self.conv4(x3))
        if self.skip_connection:
            x4 = x4 + x2

        x4 = F.interpolate(x4, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x5 = self.lrelu(self.conv5(x4))
        if self.skip_connection:
            x5 = x5 + x1

        x5 = F.interpolate(x5, size=x0.shape[2:], mode="bilinear", align_corners=False)
        x6 = self.lrelu(self.conv6(x5))
        if self.skip_connection:
            x6 = x6 + x0

        out = self.lrelu(self.conv7(x6))
        out = self.lrelu(self.conv8(out))
        out = self.conv9(out)
        return out


def build_discriminator(cfg: Config) -> nn.Module:
    """Factory to build discriminator from config.discriminator_type."""

    disc_type = str(getattr(cfg, "discriminator_type", "patchgan")).lower()
    if disc_type == "patchgan":
        return PatchGANDiscriminator(num_in_ch=cfg.num_in_ch, num_feat=cfg.disc_num_feat)
    if disc_type == "unet":
        return UNetDiscriminatorSN(num_in_ch=cfg.num_in_ch, num_feat=cfg.disc_num_feat)
    raise ValueError(f"Unsupported discriminator_type: {cfg.discriminator_type}. Use 'patchgan' or 'unet'.")
