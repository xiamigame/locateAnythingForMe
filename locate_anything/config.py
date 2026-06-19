"""
全局配置管理
"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocateConfig:
    """locateAnythingForMe 配置"""

    # 模型路径（HuggingFace model id 或本地路径）
    model_path: str = "NVEagle/Eagle-X5-13B-Chat"

    # 推理设备
    device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"

    # 数据类型
    torch_dtype: str = "float16"

    # 生成参数
    temperature: float = 0.2
    top_p: float = 0.5
    max_new_tokens: int = 512
    num_beams: int = 1
    do_sample: bool = True

    # 对话模式
    conv_mode: str = "vicuna_v1"

    # 是否使用缓存
    use_cache: bool = True


# 默认全局配置
_default_config: Optional[LocateConfig] = None


def get_default_config() -> LocateConfig:
    """获取默认配置"""
    global _default_config
    if _default_config is None:
        _default_config = LocateConfig()
    return _default_config


def set_default_config(config: LocateConfig) -> None:
    """设置默认配置"""
    global _default_config
    _default_config = config
