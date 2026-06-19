"""
locateAnythingForMe 核心 API —— 基于 NVlabs/Eagle 的 Embodied (LocateAnything-3B)
"""
import sys
from typing import List, Optional, Union
from pathlib import Path

import torch
from PIL import Image

# 将 Embodied 目录加入 sys.path，以便导入 locateanything_worker
_EMBODIED_PATH = Path(__file__).resolve().parent.parent / "submodules" / "Eagle" / "Embodied"
if str(_EMBODIED_PATH) not in sys.path:
    sys.path.insert(0, str(_EMBODIED_PATH))

from locateanything_worker import LocateAnythingWorker

from .config import LocateConfig, get_default_config


class LocateAnything:
    """
    基于 LocateAnything-3B 的视觉定位 API。

    使用示例:
        la = LocateAnything()
        result = la.detect("image.jpg", ["person", "car"])
        for box in result.boxes:
            print(box)
    """

    def __init__(self, config: Optional[LocateConfig] = None, **kwargs):
        """
        Args:
            config: LocateConfig，为 None 时使用默认配置。
            **kwargs: 覆盖 config 字段，如 model_path="...", device="cuda"。
        """
        if config is None:
            config = get_default_config()
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)

        if config.device == "cuda" and not torch.cuda.is_available():
            print("[locateAnything] CUDA 不可用，回退到 CPU")
            config.device = "cpu"

        self.config = config
        self._worker: Optional[LocateAnythingWorker] = None

    # ── 懒加载 worker ──────────────────────────────────────

    @property
    def worker(self) -> LocateAnythingWorker:
        if self._worker is None:
            print(f"[locateAnything] 加载模型: {self.config.model_path}")
            self._worker = LocateAnythingWorker(
                model_path=self.config.model_path,
                device=self.config.device,
                dtype=getattr(torch, self.config.torch_dtype),
            )
            print("[locateAnything] 模型加载完成")
        return self._worker

    # ── 检测 ────────────────────────────────────────────────

    def detect(
        self,
        image: Union[str, Image.Image],
        categories: List[str],
        **kwargs,
    ) -> "LocateResult":
        """目标检测：在图像中检测指定类别的所有实例。

        Args:
            image: 图像路径或 PIL.Image。
            categories: 类别列表，如 ["person", "car", "bicycle"]。

        Returns:
            LocateResult，包含 raw_output 和解析后的 boxes。
        """
        img = self._load_image(image)
        r = self.worker.detect(img, categories, **kwargs)
        return self._build_result(img, r)

    # ── 短语定位 ────────────────────────────────────────────

    def ground(
        self,
        image: Union[str, Image.Image],
        phrase: str,
        multi: bool = True,
        **kwargs,
    ) -> "LocateResult":
        """短语定位：根据自然语言描述定位目标。

        Args:
            image: 图像路径或 PIL.Image。
            phrase: 描述，如 "people wearing red shirts"。
            multi: True=多个实例, False=单个实例。

        Returns:
            LocateResult。
        """
        img = self._load_image(image)
        if multi:
            r = self.worker.ground_multi(img, phrase, **kwargs)
        else:
            r = self.worker.ground_single(img, phrase, **kwargs)
        return self._build_result(img, r)

    # ── 文本检测 ────────────────────────────────────────────

    def detect_text(self, image: Union[str, Image.Image], **kwargs) -> "LocateResult":
        """检测图像中的所有文字区域。"""
        img = self._load_image(image)
        r = self.worker.detect_text(img, **kwargs)
        return self._build_result(img, r)

    def ground_text(
        self, image: Union[str, Image.Image], phrase: str, **kwargs
    ) -> "LocateResult":
        """定位指定文字。"""
        img = self._load_image(image)
        r = self.worker.ground_text(img, phrase, **kwargs)
        return self._build_result(img, r)

    # ── GUI 定位 ────────────────────────────────────────────

    def ground_gui(
        self,
        image: Union[str, Image.Image],
        phrase: str,
        output_type: str = "box",
        **kwargs,
    ) -> "LocateResult":
        """GUI 元素定位（按钮、搜索框等）。

        Args:
            image: 截图路径或 PIL.Image。
            phrase: 元素描述，如 "the search button"。
            output_type: "box" 或 "point"。
        """
        img = self._load_image(image)
        r = self.worker.ground_gui(img, phrase, output_type=output_type, **kwargs)
        return self._build_result(img, output_type=output_type)

    # ── 指向 ────────────────────────────────────────────────

    def point(
        self, image: Union[str, Image.Image], phrase: str, **kwargs
    ) -> "LocateResult":
        """指向指定目标，返回点坐标。"""
        img = self._load_image(image)
        r = self.worker.point(img, phrase, **kwargs)
        return self._build_result(img, output_type="point")

    # ── 原始预测 ────────────────────────────────────────────

    def predict(
        self,
        image: Union[str, Image.Image],
        question: str,
        **kwargs,
    ) -> "LocateResult":
        """原始预测接口，自定义 prompt。"""
        img = self._load_image(image)
        r = self.worker.predict(img, question, **kwargs)
        return self._build_result(img)

    # ── 内部 ────────────────────────────────────────────────

    def _load_image(self, image: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image

    def _build_result(
        self,
        img: Image.Image,
        raw: dict,
        output_type: str = "box",
    ) -> "LocateResult":
        w, h = img.size
        answer = raw.get("answer", "")
        if output_type == "point":
            parsed = self.worker.parse_points(answer, w, h)
            return LocateResult(raw_output=answer, points=parsed, image_size=(w, h))
        else:
            parsed = self.worker.parse_boxes(answer, w, h)
            return LocateResult(raw_output=answer, boxes=parsed, image_size=(w, h))


class LocateResult:
    """定位结果"""

    def __init__(
        self,
        raw_output: str,
        boxes: Optional[List[dict]] = None,
        points: Optional[List[dict]] = None,
        image_size: Optional[tuple] = None,
    ):
        self.raw_output = raw_output
        self.boxes = boxes or []
        self.points = points or []
        self.image_size = image_size

    def __repr__(self):
        n = len(self.boxes) or len(self.points)
        return f"<LocateResult: {n} items>"

    def __str__(self):
        return self.raw_output
