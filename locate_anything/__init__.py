"""
locateAnythingForMe — 基于 LocateAnything-3B 的视觉定位 API
"""
# ── 统一环境初始化（所有 local import 之前）───────
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HOME"] = str(_PROJECT_ROOT / ".cache" / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_PROJECT_ROOT / ".cache" / "huggingface" / "hub")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

from .tool import LocateAnythingForMe, _DEFAULT_COLORS
from .api import LocateAnything  # 向后兼容别名
from .config import LocateConfig, get_default_config, set_default_config
from .video import (
    VideoLocator,
    frame_processor,
    iter_video,
    iter_camera,
    iter_screen,
    VIDEO_EXTS,
)

__version__ = "0.4.0"
__all__ = [
    "LocateAnythingForMe",
    "LocateAnything",
    "LocateConfig",
    "get_default_config",
    "set_default_config",
    "_DEFAULT_COLORS",
    "VideoLocator",
    "frame_processor",
    "iter_video",
    "iter_camera",
    "iter_screen",
    "VIDEO_EXTS",
]
