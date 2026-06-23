from __future__ import annotations

import copy
import os
import shutil
import time
from dataclasses import asdict
from typing import Any, Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor, nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import Config
from .losses import GANLoss, L1PixelLoss, PerceptualLoss
from .metrics import calculate_lpips, calculate_psnr, calculate_ssim
from .utils import autocast_ctx


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float = 0.999) -> None:
    """Update EMA model parameters from the current generator weights."""

    for p_ema, p in zip(ema_model.parameters(), model.parameters()):
        p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)


class RealESRGANFineTunerCA:
    """Trainer for CA-enhanced RealESRGAN fine-tuning with optional GAN phase."""

    def __init__(
        self,
        cfg: Config,
        net_g: nn.Module,
        net_d: nn.Module,
        net_g_ema: nn.Module,
        device: torch.device,
    ) -> None:
        """Initialize losses, optimizers, schedulers, scalers, and TensorBoard writer."""

        self.cfg = cfg
        self.device = device
        self.net_g = net_g
        self.net_d = net_d
        self.net_g_ema = net_g_ema

        self.use_amp = device.type == "cuda"

        self.cri_pixel = L1PixelLoss().to(device)
        self.cri_perceptual = PerceptualLoss(feature_layer=35).to(device)
        self.cri_gan = GANLoss().to(device)

        self.lpips_fn = None
        try:
            from .metrics import build_lpips_model

            self.lpips_fn = build_lpips_model(device)
            print("LPIPS model loaded.")
        except Exception as exc:
            print(f"LPIPS unavailable ({exc}). Validation LPIPS will be skipped.")

        ca_params = [p for n, p in self.net_g.named_parameters() if ("ca_" in n or ".ca." in n)]
        conv_params = [p for n, p in self.net_g.named_parameters() if not ("ca_" in n or ".ca." in n)]
        self.opt_g = optim.Adam(
            [{"params": conv_params, "lr": cfg.lr_g}, {"params": ca_params, "lr": cfg.lr_g_ca}],
            betas=cfg.betas,
        )
        self.opt_d = optim.Adam(self.net_d.parameters(), lr=cfg.lr_d, betas=cfg.betas)

        self.sched_g = optim.lr_scheduler.CosineAnnealingLR(self.opt_g, T_max=cfg.total_epochs, eta_min=cfg.lr_min)
        self.sched_d = optim.lr_scheduler.CosineAnnealingLR(self.opt_d, T_max=cfg.total_epochs, eta_min=cfg.lr_min)

        self.scaler_g = GradScaler(enabled=self.use_amp)
        self.scaler_d = GradScaler(enabled=self.use_amp)

        self._init_tensorboard(cfg)

        self.epoch = 0
        self.global_step = 0
        self.best_psnr = 0.0
        self.best_lpips = float("inf")
        self.best_percep = self.best_lpips
        self.discriminator_frozen = False

    def _init_tensorboard(self, cfg: Config) -> None:
        """Create SummaryWriter and optionally carry over past event files."""

        if cfg.prev_log_dir and os.path.isdir(cfg.prev_log_dir):
            for fname in os.listdir(cfg.prev_log_dir):
                src = os.path.join(cfg.prev_log_dir, fname)
                dst = os.path.join(cfg.log_dir, fname)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
            print(f"Copied previous TensorBoard events from {cfg.prev_log_dir}")
        self.writer = SummaryWriter(cfg.log_dir, purge_step=None)

    @staticmethod
    def _grad_norm(model: nn.Module) -> float:
        """Compute global L2 gradient norm for a model."""

        total = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    @staticmethod
    def _resolve_ckpt_path(path_or_name: str, checkpoint_dir: str) -> str:
        """Resolve checkpoint input as absolute path, relative path, or checkpoint-dir filename."""

        if os.path.isabs(path_or_name) and os.path.isfile(path_or_name):
            return path_or_name
        if os.path.isfile(path_or_name):
            return path_or_name
        candidate = os.path.join(checkpoint_dir, path_or_name)
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"Checkpoint not found: '{path_or_name}' (also tried '{candidate}')")

    def set_discriminator_frozen(self, freeze: bool = True, reason: str = "") -> None:
        """Freeze or unfreeze discriminator parameters and set train/eval mode."""

        if freeze == self.discriminator_frozen:
            return
        for p in self.net_d.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.net_d.eval()
        else:
            self.net_d.train()
        self.discriminator_frozen = freeze
        state = "FROZEN" if freeze else "UNFROZEN"
        extra = f" ({reason})" if reason else ""
        print(f">>> Discriminator {state}{extra}")

    def _should_freeze_discriminator(self, epoch: int, use_gan: bool) -> bool:
        """Return whether discriminator should be frozen for the current epoch."""

        if not use_gan:
            return True
        if getattr(self.cfg, "freeze_discriminator_always", False):
            return True

        start = getattr(self.cfg, "freeze_discriminator_start_epoch", None)
        end = getattr(self.cfg, "freeze_discriminator_end_epoch", None)
        if start is None or end is None:
            return False

        epoch_1based = epoch + 1
        return bool(start <= epoch_1based <= end)

    def train_step(
        self,
        lr_imgs: Tensor,
        hr_imgs: Tensor,
        use_gan: bool,
        gan_lambda: float = 0.0,
        accum_steps: int = 1,
        accum_index: int = 0,
        do_optimizer_step: bool = True,
    ) -> tuple[dict[str, float], Tensor]:
        """Run one training step with optional discriminator update and grad accumulation."""

        lr_imgs = lr_imgs.to(self.device, non_blocking=True)
        hr_imgs = hr_imgs.to(self.device, non_blocking=True)

        d_loss_val = 0.0
        if use_gan and not self.discriminator_frozen:
            self.opt_d.zero_grad(set_to_none=True)
            with torch.no_grad():
                sr_imgs = self.net_g(lr_imgs)
            with autocast_ctx(self.device, self.use_amp):
                d_real = self.net_d(hr_imgs)
                d_fake = self.net_d(sr_imgs.detach())
                d_loss = self.cri_gan(d_real, d_fake, is_disc=True)
            d_loss_val = float(d_loss.item())
            self.scaler_d.scale(d_loss).backward()
            self.scaler_d.step(self.opt_d)
            self.scaler_d.update()

        if accum_index == 0:
            self.opt_g.zero_grad(set_to_none=True)

        with autocast_ctx(self.device, self.use_amp):
            sr_imgs = self.net_g(lr_imgs)
            l_pixel = self.cri_pixel(sr_imgs, hr_imgs) * self.cfg.lambda_pixel
            l_percep = self.cri_perceptual(sr_imgs, hr_imgs) * self.cfg.lambda_perceptual
            if use_gan:
                d_real = self.net_d(hr_imgs).detach()
                d_fake = self.net_d(sr_imgs)
                l_gan = self.cri_gan(d_real, d_fake, is_disc=False) * gan_lambda
            else:
                l_gan = torch.zeros(1, device=self.device).squeeze()
            g_loss = l_pixel + l_percep + l_gan

        self.scaler_g.scale(g_loss / accum_steps).backward()

        g_grad_norm = 0.0
        if do_optimizer_step:
            self.scaler_g.unscale_(self.opt_g)
            g_grad_norm = self._grad_norm(self.net_g)
            if g_grad_norm != g_grad_norm or g_grad_norm == float("inf"):
                g_grad_norm = 0.0
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), max_norm=5.0)
            self.scaler_g.step(self.opt_g)
            self.scaler_g.update()
            update_ema(self.net_g_ema, self.net_g, self.cfg.ema_decay)

        return {
            "d_loss": d_loss_val,
            "g_loss": float(g_loss.item()),
            "pixel": float(l_pixel.item()),
            "perceptual": float(l_percep.item()),
            "gan": float(l_gan.item()) if use_gan else 0.0,
            "g_grad_norm": float(g_grad_norm),
            "gan_lambda": gan_lambda if use_gan else 0.0,
        }, sr_imgs.detach()

    @torch.no_grad()
    def validate(self, val_loader: DataLoader[Any]) -> dict[str, float]:
        """Evaluate EMA generator on validation set using PSNR, SSIM, and LPIPS."""

        self.net_g_ema.eval()
        psnr_sum, ssim_sum, lpips_sum, n = 0.0, 0.0, 0.0, 0
        for lr, hr, _ in val_loader:
            lr = lr.to(self.device, non_blocking=True)
            hr = hr.to(self.device, non_blocking=True)
            with autocast_ctx(self.device, self.use_amp):
                sr = self.net_g_ema(lr).clamp(0, 1)
            h, w = hr.shape[2:]
            sr = sr[:, :, :h, :w]
            psnr_sum += calculate_psnr(sr, hr)
            ssim_sum += calculate_ssim(sr, hr)
            if self.lpips_fn is not None:
                lpips_sum += calculate_lpips(sr, hr, self.lpips_fn)
            n += 1
        return {
            "psnr": psnr_sum / max(n, 1),
            "ssim": ssim_sum / max(n, 1),
            "lpips": lpips_sum / max(n, 1) if self.lpips_fn is not None else float("nan"),
        }

    @torch.no_grad()
    def _log_images(self, val_loader: DataLoader[Any], epoch: int) -> None:
        """Log LR, SR, and HR triplets to TensorBoard for visual monitoring."""

        self.net_g_ema.eval()
        lr, hr, _ = next(iter(val_loader))
        lr = lr.to(self.device)
        hr = hr.to(self.device)
        with autocast_ctx(self.device, self.use_amp):
            sr = self.net_g_ema(lr).clamp(0, 1)
        h, w = hr.shape[2:]
        sr = sr[:, :, :h, :w]
        lr_up = F.interpolate(lr, size=(h, w), mode="bilinear", align_corners=False)
        grid = torch.cat([lr_up, sr, hr], dim=3)
        self.writer.add_images("Images/LR_SR_HR", grid, epoch)

    def train(self, train_loader: DataLoader[Any], val_loader: Optional[DataLoader[Any]] = None) -> None:
        """Execute full training loop including validation and checkpointing."""

        cfg = self.cfg
        total_start = time.time()
        print(f"Starting CA fine-tuning for {cfg.total_epochs} epochs ...")

        for epoch in range(self.epoch, cfg.total_epochs):
            epoch_start = time.time()
            self.net_g.train()
            use_gan = epoch >= cfg.gan_start_epoch
            gan_lambda = float(cfg.lambda_gan) if use_gan else 0.0

            freeze_d = self._should_freeze_discriminator(epoch, use_gan)
            self.set_discriminator_frozen(freeze_d, reason=f"epoch {epoch + 1}")

            epoch_losses = {k: 0.0 for k in ("d_loss", "g_loss", "pixel", "perceptual", "gan", "g_grad_norm")}
            grad_norm_steps = 0
            d_loss_steps = 0

            pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"Epoch {epoch + 1}/{cfg.total_epochs}",
            )
            for batch_idx, (lr, hr, _) in pbar:
                accum_steps = max(1, int(cfg.grad_accum_steps))
                accum_index = batch_idx % accum_steps
                group_start = batch_idx - accum_index
                group_size = min(accum_steps, len(train_loader) - group_start)
                do_optimizer_step = accum_index == group_size - 1

                losses, _ = self.train_step(
                    lr,
                    hr,
                    use_gan,
                    gan_lambda=gan_lambda,
                    accum_steps=group_size,
                    accum_index=accum_index,
                    do_optimizer_step=do_optimizer_step,
                )

                if do_optimizer_step:
                    grad_norm_steps += 1
                    if use_gan and not freeze_d:
                        d_loss_steps += 1
                    for k in epoch_losses:
                        epoch_losses[k] += losses[k]

                pbar.set_postfix(
                    D=f"{losses['d_loss']:.4f}",
                    G=f"{losses['g_loss']:.4f}",
                    Pix=f"{losses['pixel']:.4f}",
                    Gl=f"{losses['gan_lambda']:.4f}",
                )

                if do_optimizer_step:
                    self.global_step += 1
                    if self.global_step % cfg.log_interval == 0:
                        for k in ("g_loss", "d_loss", "pixel", "perceptual", "gan"):
                            self.writer.add_scalar(f"Train/{k}", losses[k], self.global_step)
                        self.writer.add_scalar("Train/gan_lambda", losses["gan_lambda"], self.global_step)
                        self.writer.add_scalar("Gradients/Generator_norm", losses["g_grad_norm"], self.global_step)
                        self.writer.add_scalar("LR/generator_conv", self.opt_g.param_groups[0]["lr"], self.global_step)
                        self.writer.add_scalar("LR/generator_ca", self.opt_g.param_groups[1]["lr"], self.global_step)
                        self.writer.add_scalar("LR/discriminator", self.opt_d.param_groups[0]["lr"], self.global_step)

            for k in ("g_loss", "pixel", "perceptual", "gan", "g_grad_norm"):
                epoch_losses[k] /= max(grad_norm_steps, 1)
            if d_loss_steps > 0:
                epoch_losses["d_loss"] /= d_loss_steps
            else:
                epoch_losses["d_loss"] = float("nan")

            self.writer.add_scalar("TrainEpoch/g_loss_avg", epoch_losses["g_loss"], epoch)
            self.writer.add_scalar("TrainEpoch/pixel_avg", epoch_losses["pixel"], epoch)
            self.writer.add_scalar("TrainEpoch/perceptual_avg", epoch_losses["perceptual"], epoch)
            self.writer.add_scalar("TrainEpoch/gan_avg", epoch_losses["gan"], epoch)
            self.writer.add_scalar("TrainEpoch/g_grad_norm_avg", epoch_losses["g_grad_norm"], epoch)
            if d_loss_steps > 0:
                self.writer.add_scalar("TrainEpoch/d_loss_avg", epoch_losses["d_loss"], epoch)

            if val_loader and (epoch + 1) % cfg.image_log_interval == 0:
                self._log_images(val_loader, epoch)

            val_metrics: Optional[dict[str, float]] = None
            if val_loader and (epoch + 1) % cfg.val_interval == 0:
                val_metrics = self.validate(val_loader)
                self.writer.add_scalar("Val/PSNR", val_metrics["psnr"], epoch)
                self.writer.add_scalar("Val/SSIM", val_metrics["ssim"], epoch)
                if not (val_metrics["lpips"] != val_metrics["lpips"]):
                    self.writer.add_scalar("Val/LPIPS", val_metrics["lpips"], epoch)

                if val_metrics["psnr"] > self.best_psnr:
                    self.best_psnr = val_metrics["psnr"]
                    self.save_checkpoint("best_psnr_model.pth", val_metrics, epoch_losses)

                if use_gan and (val_metrics["lpips"] < self.best_lpips):
                    self.best_lpips = val_metrics["lpips"]
                    self.best_percep = self.best_lpips
                    self.save_checkpoint("best_perceptual_model.pth", val_metrics, epoch_losses)

            self.sched_g.step()
            if use_gan:
                self.sched_d.step()

            if (epoch + 1) % cfg.save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pth", val_metrics, epoch_losses)

            self.epoch = epoch + 1
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1} finished in {epoch_time:.0f}s")

        total_time = time.time() - total_start
        print(f"Fine-tuning complete. Total time: {total_time / 3600:.2f} hours")
        self.writer.close()

    def save_checkpoint(
        self,
        filename: str,
        val_metrics: Optional[dict[str, float]] = None,
        epoch_losses: Optional[dict[str, float]] = None,
    ) -> None:
        """Save full training state for resume and best-model tracking."""

        path = os.path.join(self.cfg.checkpoint_dir, filename)
        ckpt: dict[str, Any] = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_psnr": self.best_psnr,
            "best_lpips": self.best_lpips,
            "best_percep": self.best_percep,
            "params": self.net_g.state_dict(),
            "params_ema": self.net_g_ema.state_dict(),
            "discriminator": self.net_d.state_dict(),
            "opt_g": self.opt_g.state_dict(),
            "opt_d": self.opt_d.state_dict(),
            "sched_g": self.sched_g.state_dict(),
            "sched_d": self.sched_d.state_dict(),
            "scaler_g": self.scaler_g.state_dict(),
            "scaler_d": self.scaler_d.state_dict(),
            "config": asdict(self.cfg),
        }
        if val_metrics:
            ckpt["val_metrics"] = val_metrics
        if epoch_losses:
            ckpt["epoch_losses"] = epoch_losses
        torch.save(ckpt, path)
        print(f"Saved: {path}")

    def _apply_config_hparams_to_optimizers_and_schedulers(self) -> None:
        """Apply current config hyperparameters after resume so config values take precedence."""

        self.opt_g.param_groups[0]["lr"] = float(self.cfg.lr_g)
        self.opt_g.param_groups[1]["lr"] = float(self.cfg.lr_g_ca)
        self.opt_d.param_groups[0]["lr"] = float(self.cfg.lr_d)

        self.opt_g.defaults["betas"] = self.cfg.betas
        self.opt_d.defaults["betas"] = self.cfg.betas
        for group in self.opt_g.param_groups:
            group["betas"] = self.cfg.betas
        for group in self.opt_d.param_groups:
            group["betas"] = self.cfg.betas

        self.sched_g.base_lrs = [pg["lr"] for pg in self.opt_g.param_groups]
        self.sched_d.base_lrs = [pg["lr"] for pg in self.opt_d.param_groups]
        self.sched_g.eta_min = float(self.cfg.lr_min)
        self.sched_d.eta_min = float(self.cfg.lr_min)

    def load_checkpoint(self, path_or_name: str, override_hparams_from_config: bool = True) -> None:
        """Load checkpoint state and optionally override resumed hparams using current config."""

        path = self._resolve_ckpt_path(path_or_name, self.cfg.checkpoint_dir)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        missing_g, _ = self.net_g.load_state_dict(ckpt["params"], strict=False)
        self.net_g_ema.load_state_dict(ckpt["params_ema"], strict=False)
        self.net_d.load_state_dict(ckpt["discriminator"])

        try:
            self.opt_g.load_state_dict(ckpt["opt_g"])
        except (ValueError, KeyError):
            print("opt_g state could not be restored (param group mismatch).")

        self.opt_d.load_state_dict(ckpt["opt_d"])
        self.sched_g.load_state_dict(ckpt["sched_g"])
        self.sched_d.load_state_dict(ckpt["sched_d"])
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_psnr = ckpt.get("best_psnr", 0.0)
        self.best_lpips = ckpt.get("best_lpips", ckpt.get("best_percep", float("inf")))
        self.best_percep = self.best_lpips

        if "scaler_g" in ckpt:
            self.scaler_g.load_state_dict(ckpt["scaler_g"])
        if "scaler_d" in ckpt:
            self.scaler_d.load_state_dict(ckpt["scaler_d"])

        if override_hparams_from_config:
            self._apply_config_hparams_to_optimizers_and_schedulers()
            print("Applied current config hyperparameters after resume (lr, betas, lr_min).")

        ca_missing = [k for k in missing_g if ("ca_" in k or ".ca." in k)]
        print(
            f"Resumed from {path} (epoch {self.epoch}, step {self.global_step}, best_psnr {self.best_psnr:.2f})"
        )
        if ca_missing:
            print(f"{len(ca_missing)} CA keys kept at init (expected for baseline checkpoints).")


def build_trainer(
    cfg: Config,
    net_g: nn.Module,
    net_d: nn.Module,
    device: torch.device,
) -> RealESRGANFineTunerCA:
    """Build trainer with a frozen EMA copy initialized from the generator."""

    net_g_ema = copy.deepcopy(net_g).eval()
    for p in net_g_ema.parameters():
        p.requires_grad = False
    return RealESRGANFineTunerCA(cfg, net_g, net_d, net_g_ema, device)
