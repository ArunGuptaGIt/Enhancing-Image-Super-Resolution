from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from .config import Config


_VALID_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


class DRealSRPairedDataset(Dataset[tuple[Tensor, Tensor, str]]):
    """Paired HR/LR dataset with synchronized geometric and color augmentations."""

    def __init__(
        self,
        hr_dir: str,
        lr_dir: str,
        hr_patch_size: int = 256,
        scale_factor: int = 4,
        augment: bool = True,
    ) -> None:
        """Initialize paired image paths and patch/augmentation settings."""

        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale_factor
        self.scale_factor = scale_factor
        self.augment = augment
        self.to_tensor = transforms.ToTensor()

        hr_paths = sorted(Path(hr_dir).glob("*"))
        hr_paths = [p for p in hr_paths if p.suffix.lower() in _VALID_EXTS]

        self.pairs: list[tuple[Path, Path]] = []
        for hr_p in hr_paths:
            lr_name = hr_p.name.replace("x4", "x1")
            lr_p = Path(lr_dir) / lr_name
            if lr_p.exists():
                self.pairs.append((hr_p, lr_p))

        print(f"Found {len(self.pairs)} paired images  (HR: {hr_dir}, LR: {lr_dir})")

    def __len__(self) -> int:
        """Return number of valid HR/LR pairs."""

        return len(self.pairs)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, str]:
        """Load one HR/LR pair, apply paired transforms, and return tensors plus filename."""

        hr_path, lr_path = self.pairs[idx]
        hr_img = Image.open(hr_path).convert("RGB")
        lr_img = Image.open(lr_path).convert("RGB")

        if self.augment:
            hr_w, hr_h = hr_img.size
            max_x = max(0, hr_w - self.hr_patch_size)
            max_y = max(0, hr_h - self.hr_patch_size)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            hr_img = hr_img.crop((x, y, x + self.hr_patch_size, y + self.hr_patch_size))
            lx = x // self.scale_factor
            ly = y // self.scale_factor
            lr_img = lr_img.crop((lx, ly, lx + self.lr_patch_size, ly + self.lr_patch_size))

            if random.random() > 0.5:
                hr_img = TF.hflip(hr_img)
                lr_img = TF.hflip(lr_img)
            if random.random() > 0.5:
                hr_img = TF.vflip(hr_img)
                lr_img = TF.vflip(lr_img)
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                hr_img = TF.rotate(hr_img, angle)
                lr_img = TF.rotate(lr_img, angle)

            if random.random() > 0.5:
                cj = transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02)
                fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = transforms.ColorJitter.get_params(
                    cj.brightness, cj.contrast, cj.saturation, cj.hue
                )

                def _apply_color_jitter(img: Image.Image) -> Image.Image:
                    for fn_id in fn_idx:
                        if fn_id == 0 and brightness_factor is not None:
                            img = TF.adjust_brightness(img, brightness_factor)
                        elif fn_id == 1 and contrast_factor is not None:
                            img = TF.adjust_contrast(img, contrast_factor)
                        elif fn_id == 2 and saturation_factor is not None:
                            img = TF.adjust_saturation(img, saturation_factor)
                        elif fn_id == 3 and hue_factor is not None:
                            img = TF.adjust_hue(img, hue_factor)
                    return img

                hr_img = _apply_color_jitter(hr_img)
                lr_img = _apply_color_jitter(lr_img)
        else:
            hr_w, hr_h = hr_img.size
            cx = max(0, (hr_w - self.hr_patch_size) // 2)
            cy = max(0, (hr_h - self.hr_patch_size) // 2)
            hr_img = hr_img.crop((cx, cy, cx + self.hr_patch_size, cy + self.hr_patch_size))
            lx = cx // self.scale_factor
            ly = cy // self.scale_factor
            lr_img = lr_img.crop((lx, ly, lx + self.lr_patch_size, ly + self.lr_patch_size))

        return self.to_tensor(lr_img), self.to_tensor(hr_img), hr_path.name


def create_dataloaders(cfg: Config) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build train and validation dataloaders from a config object."""

    train_ds = DRealSRPairedDataset(
        cfg.train_hr_dir,
        cfg.train_lr_dir,
        cfg.hr_patch_size,
        cfg.scale_factor,
        augment=True,
    )
    val_ds = DRealSRPairedDataset(
        cfg.val_hr_dir,
        cfg.val_lr_dir,
        cfg.hr_patch_size,
        cfg.scale_factor,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.persistent_workers,
        prefetch_factor=cfg.prefetch_factor,
    )

    return train_loader, val_loader
