from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import VGG19_Weights, vgg19


class VGGFeatureExtractor(nn.Module):
    """Truncated VGG19 feature extractor used for perceptual supervision."""

    def __init__(self, feature_layer: int = 35, use_input_norm: bool = True) -> None:
        """Load pretrained VGG19 and keep layers up to the requested feature index."""

        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        for p in self.features.parameters():
            p.requires_grad = False
        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: Tensor) -> Tensor:
        """Extract normalized VGG features for perceptual comparison."""

        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.features(x)


class L1PixelLoss(nn.Module):
    """L1 pixel-space reconstruction loss."""

    def __init__(self) -> None:
        """Create pixel loss module."""

        super().__init__()
        self.loss = nn.L1Loss()

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        """Compute L1 distance between super-resolved and target tensors."""

        return self.loss(sr, hr)


class PerceptualLoss(nn.Module):
    """Feature-space L1 loss computed over VGG activations."""

    def __init__(self, feature_layer: int = 35) -> None:
        """Initialize VGG feature extractor and L1 criterion."""

        super().__init__()
        self.vgg = VGGFeatureExtractor(feature_layer=feature_layer)
        self.loss = nn.L1Loss()

    def forward(self, sr: Tensor, hr: Tensor) -> Tensor:
        """Compute perceptual distance between SR and HR images."""

        return self.loss(self.vgg(sr), self.vgg(hr))


class GANLoss(nn.Module):
    """Relativistic GAN loss with BCE logits objective."""

    def __init__(self, target_real: float = 1.0, target_fake: float = 0.0) -> None:
        """Create discriminator/generator adversarial loss helper."""

        super().__init__()
        self.real_label = target_real
        self.fake_label = target_fake
        self.bce = nn.BCEWithLogitsLoss()

    def _target(self, pred: Tensor, is_real: bool) -> Tensor:
        """Build constant target tensor matching a prediction tensor shape."""

        val = self.real_label if is_real else self.fake_label
        return torch.full_like(pred, val)

    def forward(self, d_real: Tensor, d_fake: Tensor, is_disc: bool) -> Tensor:
        """Compute adversarial loss for discriminator or generator update."""

        real_logit = d_real - d_fake.mean()
        fake_logit = d_fake - d_real.mean()
        if is_disc:
            loss = (
                self.bce(real_logit, self._target(real_logit, True))
                + self.bce(fake_logit, self._target(fake_logit, False))
            ) / 2
        else:
            loss = (
                self.bce(real_logit, self._target(real_logit, False))
                + self.bce(fake_logit, self._target(fake_logit, True))
            ) / 2
        return loss
