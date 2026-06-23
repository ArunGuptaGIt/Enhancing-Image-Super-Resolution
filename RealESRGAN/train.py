from __future__ import annotations

from dataclasses import fields
from typing import Optional

from .config import Config, ensure_dirs
from .data import create_dataloaders
from .models import RRDBNetCA, build_discriminator, load_pretrained_generator_ca
from .trainer import RealESRGANFineTunerCA, build_trainer
from .utils import get_device, set_seed


def build_generator_from_config(cfg: Config):
    """Construct the RRDBNetCA generator from configuration values."""

    net_g = RRDBNetCA(
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
    return net_g


def _coerce_config(cfg: Optional[object]) -> Config:
    """Convert notebook-style config objects into a typed Config instance."""

    if cfg is None:
        return Config()
    if isinstance(cfg, Config):
        return cfg

    merged = Config()
    for f in fields(Config):
        if hasattr(cfg, f.name):
            setattr(merged, f.name, getattr(cfg, f.name))

    # Backward compatibility: if callers still define num_epochs only.
    if hasattr(cfg, "num_epochs") and not hasattr(cfg, "total_epochs"):
        merged.total_epochs = int(getattr(cfg, "num_epochs"))

    # Keep both fields aligned in the final merged config.
    merged.num_epochs = merged.total_epochs
    return merged


def run_training(cfg: Optional[object] = None) -> RealESRGANFineTunerCA:
    """Run complete training pipeline and return the initialized trainer."""

    cfg = _coerce_config(cfg)

    set_seed(42)
    ensure_dirs(cfg)
    device = get_device()
    print(f"Using device: {device}")

    train_loader, val_loader = create_dataloaders(cfg)

    net_g = build_generator_from_config(cfg)
    net_g = load_pretrained_generator_ca(net_g, cfg.pretrained_model_path, device="cpu")
    net_g = net_g.to(device)

    net_d = build_discriminator(cfg).to(device)

    trainer = build_trainer(cfg, net_g, net_d, device)

    if cfg.resume_checkpoint_path:
        trainer.load_checkpoint(cfg.resume_checkpoint_path)

    trainer.train(train_loader, val_loader)
    return trainer
