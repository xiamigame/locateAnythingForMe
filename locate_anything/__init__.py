"""
locateAnythingForMe — 基于 LocateAnything-3B 的视觉定位 API
"""
from .tool import LocateAnythingForMe, _DEFAULT_COLORS
from .api import LocateAnything  # 向后兼容别名
from .config import LocateConfig, get_default_config, set_default_config

__version__ = "0.2.0"
__all__ = [
    "LocateAnythingForMe",
    "LocateAnything",
    "LocateConfig",
    "get_default_config",
    "set_default_config",
    "_DEFAULT_COLORS",
]
