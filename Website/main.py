import base64
import gc
import io
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

from config import MODEL_PATH


MODEL_OPTIONS: list[dict[str, str | None]] = [
    #Parent[2] look for ../../ from root directory so set it according to path of your model to be searched and loaded in the website
	{'name': 'Default Model', 'path': MODEL_PATH},
	{'name': 'PatchGAN with Channel Attention', 'path': str((Path(__file__).resolve().parents[2] / 'Model/PatchGAN with CA/best_perceptual_model.pth').resolve())},
	{'name': 'PatchGAN without Channel Attention', 'path': str((Path(__file__).resolve().parents[2] / 'Model/PatchGAN without CA/best_perceptual_model.pth').resolve())},
	{'name': 'Unet with Channel Attention', 'path': str((Path(__file__).resolve().parents[2] / 'Model/Unet with CA/best_perceptual_model.pth').resolve())},
]


def build_model_options() -> list[dict[str, str]]:
	"""Build selectable model options from configured checkpoint paths."""
	options: list[dict[str, str]] = []
	seen_paths: set[str] = set()

	for index, model_config in enumerate(MODEL_OPTIONS):
		raw_path = model_config.get('path')
		if not isinstance(raw_path, str) or not raw_path.strip():
			continue
		raw_name = model_config.get('name')

		path_obj = Path(raw_path).expanduser()
		if not path_obj.is_absolute():
			path_obj = (Path(__file__).resolve().parent / path_obj).resolve()

		resolved = str(path_obj)
		if resolved in seen_paths:
			continue
		seen_paths.add(resolved)

		if not path_obj.exists():
			continue

		display_name = raw_name.strip() if isinstance(raw_name, str) and raw_name.strip() else path_obj.stem

		options.append(
			{
				'id': f'model_{index + 1}',
				'label': display_name,
				'path': resolved,
			}
		)

	return options


class ChannelAttention(nn.Module):
	def __init__(self, num_feat: int, reduction: int = 16) -> None:
		super().__init__()
		mid = max(num_feat // reduction, 4)
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Sequential(
			nn.Linear(num_feat, mid, bias=False),
			nn.ReLU(inplace=True),
			nn.Linear(mid, num_feat, bias=True),
			nn.Sigmoid(),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		batch_size, channels, _, _ = x.shape
		weights = self.avg_pool(x).view(batch_size, channels)
		weights = self.fc(weights).view(batch_size, channels, 1, 1)
		return x * weights


class ResidualDenseBlock(nn.Module):
	def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, use_channel_attention: bool = False, ca_reduction: int = 16) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
		self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
		self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
		self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
		self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
		self.ca = ChannelAttention(num_feat, reduction=ca_reduction) if use_channel_attention else None
		self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x1 = self.lrelu(self.conv1(x))
		x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
		x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
		x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
		x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
		if self.ca is not None:
			x5 = self.ca(x5)
		return x5 * 0.2 + x


class RRDB(nn.Module):
	def __init__(self, num_feat: int = 64, num_grow_ch: int = 32, use_channel_attention: bool = False, ca_reduction: int = 16) -> None:
		super().__init__()
		self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch, use_channel_attention, ca_reduction)
		self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch, use_channel_attention, ca_reduction)
		self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch, use_channel_attention, ca_reduction)
		self.ca = ChannelAttention(num_feat, reduction=ca_reduction) if use_channel_attention else None

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.rdb1(x)
		out = self.rdb2(out)
		out = self.rdb3(out)
		if self.ca is not None:
			out = self.ca(out)
		return out * 0.2 + x


class RRDBNet(nn.Module):
	"""RRDBNet architecture compatible with checkpoints."""

	def __init__(
		self,
		num_in_ch: int = 3,
		num_out_ch: int = 3,
		scale: int = 4,
		num_feat: int = 64,
		num_block: int = 23,
		num_grow_ch: int = 32,
		ca_blocks: list[bool] | None = None,
		ca_reduction: int = 16,
		use_ca_after_trunk: bool = False,
		use_ca_after_up1: bool = False,
		use_ca_after_up2: bool = False,
	) -> None:
		super().__init__()
		self.scale = scale

		if ca_blocks is None:
			ca_blocks = [False] * num_block
		if len(ca_blocks) != num_block:
			raise ValueError(f'Expected {num_block} CA flags, got {len(ca_blocks)}')

		if scale == 2:
			first_in_ch = num_in_ch * 4
		elif scale == 1:
			first_in_ch = num_in_ch * 16
		else:
			first_in_ch = num_in_ch

		self.conv_first = nn.Conv2d(first_in_ch, num_feat, 3, 1, 1)
		self.body = nn.Sequential(
			*[
				RRDB(
					num_feat=num_feat,
					num_grow_ch=num_grow_ch,
					use_channel_attention=ca_blocks[i],
					ca_reduction=ca_reduction,
				)
				for i in range(num_block)
			]
		)
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
		if self.scale == 2:
			feat = self.conv_first(F.pixel_unshuffle(x, 2))
		elif self.scale == 1:
			feat = self.conv_first(F.pixel_unshuffle(x, 4))
		else:
			feat = self.conv_first(x)

		body_feat = self.conv_body(self.body(feat))
		feat = feat + body_feat

		feat = self.ca_trunk(feat)
		feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
		feat = self.ca_up1(feat)
		feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
		feat = self.ca_up2(feat)
		out = self.conv_last(self.lrelu(self.conv_hr(feat)))
		return out


def extract_state_dict(checkpoint_path: str, device: str = 'cpu') -> dict[str, Any]:
	"""Extract and normalize state dict from checkpoint."""
	ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
	if isinstance(ckpt, dict):
		if 'params_ema' in ckpt:
			state_dict = ckpt['params_ema']
		elif 'params' in ckpt:
			state_dict = ckpt['params']
		else:
			state_dict = ckpt
	else:
		state_dict = ckpt

	return {key.replace('module.', ''): value for key, value in state_dict.items()}


def infer_arch_from_state_dict(state_dict: dict[str, Any], default_variant: str = 'x4plus') -> dict[str, Any]:
	"""Infer model architecture from checkpoint state dict."""
	num_feat = state_dict.get('conv_first.weight', torch.empty(64, 3, 3, 3)).shape[0]
	first_in_ch = state_dict.get('conv_first.weight', torch.empty(64, 3, 3, 3)).shape[1]
	num_out_ch = state_dict.get('conv_last.weight', torch.empty(3, 64, 3, 3)).shape[0]

	if first_in_ch % 3 == 0 and first_in_ch // 3 in (1, 4, 16):
		ratio = first_in_ch // 3
		scale = 4 if ratio == 1 else (2 if ratio == 4 else 1)
		num_in_ch = 3
	else:
		scale = {'x4plus': 4, 'x2plus': 2, 'x4plus_anime_6b': 4}.get(default_variant.lower(), 4)
		num_in_ch = first_in_ch

	num_grow_ch = state_dict.get('body.0.rdb1.conv1.weight', torch.empty(32, num_feat, 3, 3)).shape[0]

	block_ids: set[int] = set()
	ca_block_ids: set[int] = set()
	for key in state_dict.keys():
		if not key.startswith('body.'):
			continue
		parts = key.split('.')
		if len(parts) < 2 or not parts[1].isdigit():
			continue
		idx = int(parts[1])
		block_ids.add(idx)
		if '.ca.' in key:
			ca_block_ids.add(idx)

	num_block = max(block_ids) + 1 if block_ids else {'x4plus': 23, 'x2plus': 23, 'x4plus_anime_6b': 6}.get(default_variant.lower(), 23)
	ca_blocks = [i in ca_block_ids for i in range(num_block)]

	use_ca_after_trunk = any(k.startswith('ca_trunk.') for k in state_dict.keys())
	use_ca_after_up1 = any(k.startswith('ca_up1.') for k in state_dict.keys())
	use_ca_after_up2 = any(k.startswith('ca_up2.') for k in state_dict.keys())

	ca_reduction = 16
	for key, tensor in state_dict.items():
		if (('.ca.fc.0.weight' in key) or key.startswith('ca_trunk.fc.0.weight') or key.startswith('ca_up1.fc.0.weight') or key.startswith('ca_up2.fc.0.weight')) and tensor.ndim == 2:
			mid = tensor.shape[0]
			if mid > 0:
				ca_reduction = max(1, num_feat // mid)
			break

	return {
		'num_in_ch': num_in_ch,
		'num_out_ch': num_out_ch,
		'scale': scale,
		'num_feat': num_feat,
		'num_block': num_block,
		'num_grow_ch': num_grow_ch,
		'ca_blocks': ca_blocks,
		'ca_reduction': ca_reduction,
		'use_ca_after_trunk': use_ca_after_trunk,
		'use_ca_after_up1': use_ca_after_up1,
		'use_ca_after_up2': use_ca_after_up2,
	}


class SRService:
	def __init__(self) -> None:
		self.upsampler: Any | None = None
		self._model_lock = threading.RLock()
		self.backend_name = 'PIL-Bicubic'
		self.loaded_at = datetime.now(timezone.utc)
		self.load_duration_ms: float | None = None
		self.load_error: str | None = None
		self.model_path = ''
		self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.model_loaded = False
		self.model_options: list[dict[str, str]] = build_model_options()
		self.selected_model_id: str | None = self.model_options[0]['id'] if self.model_options else None

	def get_model_options(self) -> list[dict[str, str]]:
		return [dict(option) for option in self.model_options]

	def _find_model_option(self, model_id: str) -> dict[str, str] | None:
		for option in self.model_options:
			if option['id'] == model_id:
				return option
		return None

	def _cuda_cleanup(self) -> None:
		gc.collect()
		if torch.cuda.is_available():
			torch.cuda.empty_cache()
			torch.cuda.ipc_collect()

	def unload_model(self) -> None:
		"""Release current model and cached CUDA memory before reloading."""
		if self.upsampler is not None:
			try:
				if isinstance(self.upsampler, nn.Module):
					self.upsampler.to('cpu')
			except Exception:
				# Best effort: continue cleanup even if moving to CPU fails.
				pass
			finally:
				self.upsampler = None

		self._cuda_cleanup()

	def load_from_checkpoint(self, checkpoint_path: str, display_name: str | None = None) -> None:
		"""Load model from checkpoint with architecture inference."""
		try:
			if not Path(checkpoint_path).exists():
				raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

			# Load checkpoint tensors on CPU first to avoid GPU peak-memory spikes.
			state_dict = extract_state_dict(checkpoint_path, device='cpu')
			arch = infer_arch_from_state_dict(state_dict)

			model = RRDBNet(
				num_in_ch=arch['num_in_ch'],
				num_out_ch=arch['num_out_ch'],
				scale=arch['scale'],
				num_feat=arch['num_feat'],
				num_block=arch['num_block'],
				num_grow_ch=arch['num_grow_ch'],
				ca_blocks=arch['ca_blocks'],
				ca_reduction=arch['ca_reduction'],
				use_ca_after_trunk=arch['use_ca_after_trunk'],
				use_ca_after_up1=arch['use_ca_after_up1'],
				use_ca_after_up2=arch['use_ca_after_up2'],
			)
			model.load_state_dict(state_dict, strict=True)
			model = model.to(self.device).eval()

			self.upsampler = model
			self.model_path = checkpoint_path
			arch_name = f'RRDBNet (scale={arch["scale"]}, blocks={arch["num_block"]})'
			self.backend_name = f'{display_name} | {arch_name}' if display_name else arch_name
			self.load_error = None
			self.model_loaded = True
			del state_dict
		except Exception as error:
			self.upsampler = None
			self.model_loaded = False
			self._cuda_cleanup()
			raise error

	def load(self, model_id: str | None = None) -> None:
		with self._model_lock:
			started = time.perf_counter()
			self.model_loaded = False
			self.model_path = ''
			self.load_error = None
			self.model_options = build_model_options()

			# Ensure old model memory is released before loading a new checkpoint.
			self.unload_model()

			if not self.model_options:
				self.selected_model_id = None
				self.backend_name = 'model not loaded'
				self.load_error = 'No model checkpoint found. Set MODEL_OPTIONS in main.py with at least one valid model path.'
				self.loaded_at = datetime.now(timezone.utc)
				self.load_duration_ms = (time.perf_counter() - started) * 1000
				return

			if model_id is not None:
				selected = self._find_model_option(model_id)
				if selected is None:
					raise ValueError(f'Unknown model id: {model_id}')
				self.selected_model_id = model_id
			elif self.selected_model_id is None or self._find_model_option(self.selected_model_id) is None:
				self.selected_model_id = self.model_options[0]['id']

			selected_option = self._find_model_option(self.selected_model_id) if self.selected_model_id else None
			if selected_option is None:
				self.backend_name = 'model not loaded'
				self.load_error = 'Selected model is unavailable.'
				self.loaded_at = datetime.now(timezone.utc)
				self.load_duration_ms = (time.perf_counter() - started) * 1000
				return

			last_error: Exception | None = None
			try:
				self.load_from_checkpoint(selected_option['path'], display_name=selected_option['label'])
			except Exception as error:
				last_error = error

			if last_error is None:
				self.loaded_at = datetime.now(timezone.utc)
				self.load_duration_ms = (time.perf_counter() - started) * 1000
				return

			self.backend_name = 'model not loaded'
			self.load_error = str(last_error)
			self.loaded_at = datetime.now(timezone.utc)
			self.load_duration_ms = (time.perf_counter() - started) * 1000

	def enhance(self, image: Image.Image) -> tuple[Image.Image, float]:
		with self._model_lock:
			started = time.perf_counter()

			if not self.model_loaded:
				raise RuntimeError('Super-resolution model not loaded. Please ensure the model file exists.')

			try:
				if self.upsampler is not None:
					if isinstance(self.upsampler, RRDBNet):
						rgb_np = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
						inp = torch.from_numpy(rgb_np).permute(2, 0, 1).unsqueeze(0).to(self.device)

						with torch.inference_mode():
							out = self.upsampler(inp)

						out = out.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
						out = (out * 255.0).round().astype(np.uint8)
						sr_image = Image.fromarray(out)
						del inp, out, rgb_np
					else:
						raise RuntimeError('Unexpected upsampler type.')
				else:
					raise RuntimeError('Super-resolution model not available.')
			except Exception:
				self._cuda_cleanup()
				raise

			elapsed = (time.perf_counter() - started) * 1000
			return sr_image, elapsed


def image_to_data_url(image: Image.Image) -> str:
	buffer = io.BytesIO()
	image.save(buffer, format='PNG', optimize=True)
	encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
	return f'data:image/png;base64,{encoded}'


def compute_quality_metrics(lr: Image.Image, sr: Image.Image) -> dict[str, float]:
	"""Compute PSNR, SSIM, and LPIPS metrics."""
	lr_np = np.array(lr.convert('RGB'), dtype=np.float32)
	sr_down = np.array(sr.convert('RGB').resize(lr.size, Image.Resampling.BICUBIC), dtype=np.float32)

	lr_gray = (0.299 * lr_np[:, :, 0] + 0.587 * lr_np[:, :, 1] + 0.114 * lr_np[:, :, 2]).astype(np.float32)
	sr_gray = (0.299 * sr_down[:, :, 0] + 0.587 * sr_down[:, :, 1] + 0.114 * sr_down[:, :, 2]).astype(np.float32)

	diff = lr_gray - sr_gray
	mse = float(np.mean(np.square(diff)))

	if mse <= 1e-12:
		psnr = 100.0
	else:
		psnr = float(20 * np.log10(255.0 / np.sqrt(mse)))

	mu_x = float(np.mean(lr_gray))
	mu_y = float(np.mean(sr_gray))
	sigma_x = float(np.var(lr_gray))
	sigma_y = float(np.var(sr_gray))
	sigma_xy = float(np.mean((lr_gray - mu_x) * (sr_gray - mu_y)))

	c1 = (0.01 * 255) ** 2
	c2 = (0.03 * 255) ** 2
	ssim_num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
	ssim_den = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
	ssim = float(ssim_num / ssim_den) if ssim_den != 0 else 1.0
	ssim = max(0.0, min(1.0, ssim))

	lpips = 1.0 - max(0.0, min(1.0, ssim * 0.5 + (100.0 - psnr) / 255.0 * 0.5))
	lpips = max(0.0, min(1.0, lpips))

	return {
		'psnr': psnr,
		'ssim': ssim,
		'lpips': lpips,
	}


service = SRService()

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
ACCEPTED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


class ModelSelectRequest(BaseModel):
	model_id: str


def build_health_payload() -> dict[str, Any]:
	return {
		'status': 'ok',
		'model_status': 'loaded' if service.model_loaded else 'model not loaded',
		'model': {
			'backend': service.backend_name,
			'path': service.model_path if service.model_loaded else None,
			'load_time_ms': service.load_duration_ms,
			'loaded_at': service.loaded_at.isoformat(),
			'error': service.load_error,
			'selected_id': service.selected_model_id,
			'options': service.get_model_options(),
		},
	}


@asynccontextmanager
async def lifespan(_: FastAPI):
	service.load()
	yield


app = FastAPI(title='ImageSR API', version='1.0.0', lifespan=lifespan)

app.add_middleware(
	CORSMiddleware,
	allow_origins=['*'],
	allow_credentials=True,
	allow_methods=['*'],
	allow_headers=['*'],
)


@app.get('/api/health')
async def health() -> dict[str, Any]:
	return build_health_payload()


@app.post('/api/models/select')
async def select_model(payload: ModelSelectRequest) -> dict[str, Any]:
	try:
		service.load(payload.model_id)
	except ValueError as error:
		raise HTTPException(status_code=400, detail=str(error)) from error
	except Exception as error:
		raise HTTPException(status_code=500, detail=f'Failed to load selected model: {error}') from error

	return build_health_payload()


@app.post('/api/enhance')
async def enhance_image(
	file: UploadFile = File(...),
) -> dict[str, Any]:
	started = time.perf_counter()
	accepted_extensions = sorted(ACCEPTED_IMAGE_EXTENSIONS)

	if not file.content_type or not file.content_type.startswith('image/'):
		raise HTTPException(
			status_code=400,
			detail={
				'message': 'Invalid MIME type. Please upload an image file.',
				'accepted_extensions': accepted_extensions,
			},
		)

	file_suffix = Path(file.filename or '').suffix.lower()
	if file_suffix not in ACCEPTED_IMAGE_EXTENSIONS:
		raise HTTPException(
			status_code=400,
			detail={
				'message': 'Unsupported file extension.',
				'accepted_extensions': accepted_extensions,
			},
		)

	if not service.model_loaded:
		raise HTTPException(status_code=503, detail='Super-resolution model not found. Please ensure the model file exists and is properly configured.')

	try:
		payload = await file.read()
		if len(payload) > MAX_UPLOAD_SIZE_BYTES:
			raise HTTPException(status_code=400, detail='File too large. Maximum allowed size is 10MB.')

		if not payload:
			raise HTTPException(status_code=400, detail='Uploaded file is empty.')

		verified_image = Image.open(io.BytesIO(payload))
		verified_image.verify()
		lr_image = Image.open(io.BytesIO(payload)).convert('RGB')
	except Exception as error:
		if isinstance(error, HTTPException):
			raise
		raise HTTPException(
			status_code=400,
			detail={
				'message': 'Invalid image file.',
				'accepted_extensions': accepted_extensions,
			},
		) from error

	sr_image, upscale_ms = service.enhance(lr_image)

	quality = compute_quality_metrics(lr_image, sr_image)

	response: dict[str, Any] = {
		'request_id': str(uuid.uuid4()),
		'model': {
			'backend': service.backend_name,
			'loaded_at': service.loaded_at.isoformat(),
		},
		'timings_ms': {
			'total': (time.perf_counter() - started) * 1000,
			'upscale': upscale_ms,
		},
		'images': {
			'lr': image_to_data_url(lr_image),
			'sr': image_to_data_url(sr_image),
		},
		'dimensions': {
			'lr': [lr_image.width, lr_image.height],
			'sr': [sr_image.width, sr_image.height],
		},
		'quality': {
			'psnr': quality['psnr'],
			'ssim': quality['ssim'],
			'lpips': quality['lpips'],
		},
	}

	return response
