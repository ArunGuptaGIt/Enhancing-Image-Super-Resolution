from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image
import torch
from torch import nn
from tqdm import tqdm
import torchvision.transforms as transforms

from .config import Config
from .models import RRDBNetCA
from .utils import get_device


_VALID_EXTS: tuple[str, ...] = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _iter_image_paths(folder: str, recursive: bool = False) -> Iterable[Path]:
    """Yield supported image paths from a folder in sorted order."""
    root = Path(folder)
    if recursive:
        candidates = sorted(root.rglob("*"))
    else:
        candidates = sorted(root.glob("*"))
    for p in candidates:
        if p.is_file() and p.suffix.lower() in _VALID_EXTS:
            yield p


def _build_generator_from_config(cfg: Config) -> nn.Module:
    """Build the CA-enabled RRDB generator from configuration values."""
    return RRDBNetCA(
        num_in_ch=cfg.num_in_ch,
        num_out_ch=cfg.num_out_ch,
        scale=cfg.scale_factor,
        num_feat=cfg.num_feat,
        num_block=cfg.num_block,
        num_grow_ch=cfg.num_grow_ch,
        ca_reduction=cfg.ca_reduction,
        use_ca_after_trunk=cfg.use_ca_after_trunk,
        use_ca_after_up1=cfg.use_ca_after_up1,
        use_ca_after_up2=cfg.use_ca_after_up2,
    )


def load_finetuned_model_ca(
    weight_path: str,
    cfg: Config,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load a CA generator checkpoint and return an eval-mode model on the selected device."""
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Weights not found: {weight_path}")

    run_device = device or get_device()
    model = _build_generator_from_config(cfg)

    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, dict) and "params_ema" in checkpoint:
        state_dict = checkpoint["params_ema"]
        print(f"Loaded EMA weights from: {weight_path}")
    elif isinstance(checkpoint, dict) and "params" in checkpoint:
        state_dict = checkpoint["params"]
        print(f"Loaded params weights from: {weight_path}")
    else:
        state_dict = checkpoint
        print(f"Loaded raw state_dict from: {weight_path}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys during load: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys during load: {len(unexpected)}")

    model = model.to(run_device)
    model.eval()
    return model


def run_inference_folder(
    cfg: Config,
    weight_path: str,
    input_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    keep_structure: bool = False,
    device: Optional[torch.device] = None,
) -> list[str]:
    """Run super-resolution inference for all images in input_dir and save outputs to output_dir.

    Args:
        cfg: RealESRGAN model/config settings used to construct the network.
        weight_path: Path to checkpoint file.
        input_dir: Folder containing LR images. If None, cfg.inference_input_dir is used.
        output_dir: Folder where SR images will be written. If None, cfg.inference_output_dir is used.
        recursive: If True, scan input_dir recursively.
        keep_structure: If True and recursive is enabled, preserve relative paths.
        device: Optional torch device override.

    Returns:
        List of saved output file paths.
    """
    resolved_input_dir = input_dir or cfg.inference_input_dir
    resolved_output_dir = output_dir or cfg.inference_output_dir

    if not resolved_input_dir:
        raise ValueError("Inference input directory is empty. Set cfg.inference_input_dir or pass input_dir.")
    if not resolved_output_dir:
        raise ValueError("Inference output directory is empty. Set cfg.inference_output_dir or pass output_dir.")

    if not os.path.isdir(resolved_input_dir):
        raise FileNotFoundError(f"Input folder not found: {resolved_input_dir}")

    os.makedirs(resolved_output_dir, exist_ok=True)
    run_device = device or get_device()

    model = load_finetuned_model_ca(weight_path=weight_path, cfg=cfg, device=run_device)
    to_tensor = transforms.ToTensor()

    image_paths = list(_iter_image_paths(resolved_input_dir, recursive=recursive))
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {resolved_input_dir}")

    saved_paths: list[str] = []

    with torch.no_grad():
        for in_path in tqdm(image_paths, desc="Inference"):
            img = Image.open(in_path).convert("RGB")
            lr_tensor = to_tensor(img).unsqueeze(0).to(run_device)

            if run_device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    sr_tensor = model(lr_tensor)
            else:
                sr_tensor = model(lr_tensor)

            sr_tensor = sr_tensor.float().clamp(0, 1).squeeze(0).cpu()
            sr_np = (sr_tensor.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
            out_img = Image.fromarray(sr_np)

            if keep_structure and recursive:
                rel = Path(in_path).relative_to(Path(resolved_input_dir))
                out_path = Path(resolved_output_dir) / rel
                out_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                out_path = Path(resolved_output_dir) / Path(in_path).name

            out_img.save(out_path)
            saved_paths.append(str(out_path))

    print(f"Saved {len(saved_paths)} SR images to: {resolved_output_dir}")
    return saved_paths
