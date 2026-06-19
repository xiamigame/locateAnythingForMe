"""
全局配置 — 单例 LocateConfig 贯穿所有层级
"""
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class LocateConfig:
    """locateAnythingForMe 统一配置（模型 + 检测 + 生成）"""

    # 模型
    model_path: str = "nvidia/LocateAnything-3B"
    device: str = "cuda"
    torch_dtype: str = "float16"

    # 图像缩放
    max_edge: int = 512
    min_edge: Optional[int] = None

    # 生成参数
    generation_mode: str = "hybrid"  # "fast" | "slow" | "hybrid"
    temperature: float = 0.7         # 0=贪心解码，>0 采样
    max_new_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 0

    # 批处理
    use_batch_runtime: bool = False
    attn: str = "la_flash"
    vision_attn: str = "auto"
    scheduler: str = "pipeline"
    group_size: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def to_detect_dict(self) -> dict:
        """提取生成参数，传给模型 predict()"""
        return {
            "generation_mode": self.generation_mode,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


_default_config: Optional[LocateConfig] = None


def get_default_config() -> LocateConfig:
    if _default_config is None:
        return LocateConfig()
    return _default_config


def set_default_config(config: LocateConfig) -> None:
    global _default_config
    _default_config = config
