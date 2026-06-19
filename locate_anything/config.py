"""
全局配置管理
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocateConfig:
    """locateAnythingForMe 配置 —— 基于 LocateAnything-3B"""

    # 模型路径（HuggingFace model id 或本地路径）
    model_path: str = "nvidia/LocateAnything-3B"

    # 推理设备
    device: str = "cuda"

    # 数据类型 (bfloat16 / float16 / float32)
    torch_dtype: str = "bfloat16"

    # 生成参数
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.1
    generation_mode: str = "hybrid"  # "fast" | "slow" | "hybrid"

    # 批处理（需要下载 HF 模型仓库中的 batch_utils/ 和 kernel_utils/）
    use_batch_runtime: bool = False
    attn: str = "la_flash"
    vision_attn: str = "auto"
    scheduler: str = "pipeline"
    group_size: int = 0


# 默认全局配置
_default_config: Optional[LocateConfig] = None


def get_default_config() -> LocateConfig:
    if _default_config is None:
        return LocateConfig()
    return _default_config


def set_default_config(config: LocateConfig) -> None:
    global _default_config
    _default_config = config
