"""
locateAnythingForMe - 基于 NVlabs/Eagle 的视觉定位与多模态理解 API
"""
from .api import LocateAnything, LocateResult
from .config import LocateConfig, get_default_config, set_default_config

__version__ = "0.1.0"
__all__ = [
    "LocateAnything",
    "LocateResult",
    "LocateConfig",
    "get_default_config",
    "set_default_config",
]
