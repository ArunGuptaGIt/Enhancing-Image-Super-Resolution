from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
	"""Training and model configuration for the modular RealESRGAN pipeline.

	This dataclass centralizes paths, architecture choices, optimizer settings,
	and training schedule values consumed by the package entrypoint.
	"""

	pretrained_model_path: str = ""
	resume_checkpoint_path: Optional[str] = None

	train_hr_dir: str = ""
	train_lr_dir: str = ""
	val_hr_dir: str = ""
	val_lr_dir: str = ""
	inference_input_dir: str = ""
	inference_output_dir: str = "imageoutputSRGenerated"

	checkpoint_dir: str = "experiments/checkpoints"
	log_dir: str = "experiments/logs"
	prev_log_dir: Optional[str] = None

	num_in_ch: int = 3
	num_out_ch: int = 3
	scale_factor: int = 4
	num_feat: int = 64
	num_block: int = 23
	num_grow_ch: int = 32

	ca_reduction: int = 8
	use_ca_after_trunk: bool = True
	use_ca_after_up1: bool = True
	use_ca_after_up2: bool = True

	disc_num_feat: int = 64
	discriminator_type: str = "patchgan"
	freeze_discriminator_always: bool = False
	freeze_discriminator_start_epoch: Optional[int] = None
	freeze_discriminator_end_epoch: Optional[int] = None

	total_epochs: int = 18
	num_epochs: int = 18
	hr_patch_size: int = 192
	lr_patch_size: int = 48

	batch_size: int = 8
	grad_accum_steps: int = 2

	num_workers: int = 4
	persistent_workers: bool = True
	prefetch_factor: int = 4

	gan_start_epoch: int = 10

	lr_g: float = 5e-6
	lr_g_ca: float = 1e-4
	lr_d: float = 1e-5
	betas: tuple[float, float] = (0.9, 0.99)
	lr_min: float = 5e-8

	lambda_pixel: float = 1.0
	lambda_perceptual: float = 0.8
	lambda_gan: float = 0.1

	ema_decay: float = 0.9995

	save_interval: int = 3
	val_interval: int = 1
	log_interval: int = 30
	image_log_interval: int = 3


def ensure_dirs(cfg: Config) -> None:
	"""Create filesystem directories required for checkpoints and logs."""

	import os

	os.makedirs(cfg.checkpoint_dir, exist_ok=True)
	os.makedirs(cfg.log_dir, exist_ok=True)

