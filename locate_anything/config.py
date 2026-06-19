"""
全局配置
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LocateConfig:
    """locateAnythingForMe 配置"""

    # 模型
    model_path: str = "nvidia/LocateAnything-3B"
    device: str = "cuda"
    torch_dtype: str = "float16"  # bfloat16 / float16 / float32（RTX 3080 10G 用 float16）

    # 图像缩放策略
    max_edge: int = 512        # 最长边缩放到此值（512=RTX3080安全, 768=较快, 1024=需大显存）
    min_edge: Optional[int] = None  # 最短边最小值（None=不限制）

    # 生成参数
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    max_new_tokens: int = 2048
    repetition_penalty: float = 1.1
    generation_mode: str = "hybrid"  # "fast" | "slow" | "hybrid"

    # 批处理
    use_batch_runtime: bool = False
    attn: str = "la_flash"
    vision_attn: str = "auto"
    scheduler: str = "pipeline"
    group_size: int = 0


_default_config: Optional[LocateConfig] = None


def get_default_config() -> LocateConfig:
    if _default_config is None:
        return LocateConfig()
    return _default_config


def set_default_config(config: LocateConfig) -> None:
    global _default_config
    _default_config = config
