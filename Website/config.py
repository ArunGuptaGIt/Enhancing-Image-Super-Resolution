"""Configuration for Image SR Backend

Set your custom model path here. Relative paths are resolved from the project root.
Examples:
    - "../RealESRGAN_x4plus.pth"
    - "../output(16-26)CA_Unet/kaggle/working/experiments/checkpoints/best_perceptual_model.pth"
    - None (uses automatic detection)
"""

from pathlib import Path

# Set your model path here - leave None for automatic detection
# MODEL_PATH = None  # Auto-detect from workspace
MODEL_PATH = "../output(16-26)CA_Unet/kaggle/working/experiments/checkpoints/best_perceptual_model.pth"

# Convert to absolute path if relative
if MODEL_PATH:
    model_path_obj = Path(MODEL_PATH)
    if not model_path_obj.is_absolute():
        model_path_obj = (Path(__file__).parent.parent / MODEL_PATH).resolve()
    MODEL_PATH = str(model_path_obj)
