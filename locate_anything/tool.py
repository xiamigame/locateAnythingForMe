"""
LocateAnythingForMe —— 内部封装工具

在官方 LocateAnythingWorker 基础上增加：
- 智能图像缩放（等比缩放到最长边 max_edge，降低显存并加速推理）
- 坐标自动映射回原图尺寸
- 检测 + 标注一体化
"""
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image, ImageDraw, ImageFont

_log = logging.getLogger("tool")

# ── 导入官方 worker ────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_EMBODIED_PATH = _PROJECT_ROOT / "submodules" / "Eagle" / "Embodied"
if str(_EMBODIED_PATH) not in sys.path:
    sys.path.insert(0, str(_EMBODIED_PATH))

from locateanything_worker import LocateAnythingWorker


# ── 默认类别配色 ─────────────────────────────────────────
_DEFAULT_COLORS = [
    (220, 20, 60),   # crimson
    (0, 180, 0),     # green
    (0, 120, 255),   # blue
    (255, 165, 0),   # orange
    (160, 32, 240),  # purple
    (0, 200, 200),   # cyan
    (255, 50, 150),  # pink
    (180, 180, 0),   # olive
    (100, 80, 255),  # indigo
    (0, 150, 100),   # teal
]


class LocateAnythingForMe(LocateAnythingWorker):
    """
    继承官方 LocateAnythingWorker，增加图像预处理和坐标映射。

    使用示例:
        la = LocateAnythingForMe(config=LocateConfig(max_edge=1024))
        result = la.detect("photo.jpg", ["person", "car"])
        annotated = la.annotate("photo.jpg", result)
        annotated.save("output.jpg")
    """

    def __init__(
        self,
        model_path: str = "nvidia/LocateAnything-3B",
        device: str = "cuda",
        dtype=torch.bfloat16,
        *,
        config: Optional["LocateConfig"] = None,
        **worker_kwargs,
    ):
        """
        Args:
            model_path: HuggingFace 模型 ID（config 提供时忽略）。
            device: 推理设备（config 提供时忽略）。
            dtype: 模型数据类型（config 提供时忽略）。
            config: LocateConfig 实例，包含所有模型/缩放/生成参数。
            **worker_kwargs: 传给 LocateAnythingWorker 的额外参数。
        """
        from .config import LocateConfig as _LocateConfig

        if config is None:
            config = _LocateConfig(model_path=model_path, device=device)

        self.config = config

        super().__init__(
            model_path=config.model_path,
            device=config.device,
            dtype=dtype,
            **worker_kwargs,
        )

    @property
    def max_edge(self):
        return self.config.max_edge

    @max_edge.setter
    def max_edge(self, value):
        self.config.max_edge = value

    @property
    def min_edge(self):
        return self.config.min_edge

    @min_edge.setter
    def min_edge(self, value):
        self.config.min_edge = value

    # ── 图像缩放 ──────────────────────────────────────────

    def _smart_resize(
        self, image: Image.Image
    ) -> tuple[Image.Image, float, tuple[int, int]]:
        """
        等比缩放图像，使最长边不超过 self.max_edge。

        Args:
            image: 原始 PIL Image。

        Returns:
            (resized_image, scale_factor, original_size)
            - resized_image: 缩放后的图像
            - scale_factor: 原图相对于缩放图的倍数（>1 表示原图更大）
            - original_size: (w, h) 原始尺寸
        """
        orig_w, orig_h = image.size
        longest = max(orig_w, orig_h)

        if longest <= self.max_edge:
            _log.info("[tool] resize skip: %dx%d ≤ %d", orig_w, orig_h, self.max_edge)
            return image, 1.0, (orig_w, orig_h)

        t0 = time.time()
        scale = self.max_edge / longest
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 保证最短边不小于 min_edge（如果设置了）
        if self.min_edge and min(new_w, new_h) < self.min_edge:
            scale = self.min_edge / min(orig_w, orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

        resized = image.resize((new_w, new_h), Image.LANCZOS)
        scale_factor = 1.0 / scale  # 原图 / 缩放图 的倍率
        _log.info("[tool] resize: %dx%d → %dx%d | %.0fms",
                   orig_w, orig_h, new_w, new_h, (time.time() - t0) * 1000)
        return resized, scale_factor, (orig_w, orig_h)

    # ── 坐标映射（输出坐标 ∈ [0, 1000]）────────────────────

    @staticmethod
    def _map_boxes_back(
        raw_answer: str,
        scale_factor: float,
    ) -> str:
        """将模型输出中的 box 坐标从缩放图空间映射回原图空间。

        模型输出坐标是 [0, 1000] 的归一化值，通过图片实际像素换算。
        缩放后的图片改变了实际像素，但模型输出仍然是基于缩放图的像素。

        Actually: 模型输出坐标范围 [0, 1000] 是归一化的，与图片实际像素无关。
        我们直接在 parse 时用原图尺寸计算即可，不需要改 answer 字符串。
        """
        return raw_answer

    @staticmethod
    def parse_boxes_scaled(
        answer: str, original_width: int, original_height: int
    ) -> list[dict]:
        """用原图尺寸解析 box + 规范化坐标（确保 x1<=x2, y1<=y2）。"""
        boxes = LocateAnythingWorker.parse_boxes(answer, original_width, original_height)
        return LocateAnythingForMe._normalize_boxes(boxes)

    @staticmethod
    def parse_points_scaled(
        answer: str, original_width: int, original_height: int
    ) -> list[dict]:
        """用原图尺寸解析 point。"""
        return LocateAnythingWorker.parse_points(answer, original_width, original_height)

    @staticmethod
    def _normalize_boxes(boxes: list[dict]) -> list[dict]:
        """规范化 box 坐标：确保 x1<=x2, y1<=y2, 裁剪到非负。"""
        for b in boxes:
            if b["x1"] > b["x2"]:
                b["x1"], b["x2"] = b["x2"], b["x1"]
            if b["y1"] > b["y2"]:
                b["y1"], b["y2"] = b["y2"], b["y1"]
            b["x1"] = max(0, b["x1"])
            b["y1"] = max(0, b["y1"])
            b["x2"] = max(0, b["x2"])
            b["y2"] = max(0, b["y2"])
        return boxes

    # ── 封装推理方法（resize → infer → parse on original）───

    def detect(
        self,
        image: Union[str, Image.Image],
        categories: list[str],
        **kwargs,
    ) -> dict:
        """目标检测 —— 自动缩放 + 坐标用原图解析。

        Returns:
            dict with keys:
                - answer: 原始模型输出文本
                - boxes: 解析后的框列表（原图像素坐标）
                - original_size: (w, h)
                - resized_size: (w, h)
                - scale_factor: 缩放倍率
        """
        img = self._load(image)
        img_small, scale, (orig_w, orig_h) = self._smart_resize(img)

        t0 = time.time()
        raw = super().detect(img_small, categories, **{**self.config.to_detect_dict(), **kwargs})
        infer_ms = (time.time() - t0) * 1000

        boxes = self.parse_boxes_scaled(raw["answer"], orig_w, orig_h)
        _log.info("[tool] detect | %.0fms | %d boxes | %dx%d→%dx%d",
                  infer_ms, len(boxes), orig_w, orig_h, *img_small.size)

        return {
            "answer": raw["answer"],
            "boxes": boxes,
            "original_size": (orig_w, orig_h),
            "resized_size": img_small.size,
            "scale_factor": scale,
        }

    def ground_multi(
        self,
        image: Union[str, Image.Image],
        phrase: str,
        **kwargs,
    ) -> dict:
        """多实例短语定位。"""
        img = self._load(image)
        img_small, scale, (orig_w, orig_h) = self._smart_resize(img)

        t0 = time.time()
        raw = super().ground_multi(img_small, phrase, **{**self.config.to_detect_dict(), **kwargs})
        infer_ms = (time.time() - t0) * 1000

        boxes = self.parse_boxes_scaled(raw["answer"], orig_w, orig_h)
        _log.info("[tool] ground_multi | %.0fms | %d boxes | %dx%d→%dx%d",
                  infer_ms, len(boxes), orig_w, orig_h, *img_small.size)

        return {
            "answer": raw["answer"],
            "boxes": boxes,
            "original_size": (orig_w, orig_h),
            "resized_size": img_small.size,
            "scale_factor": scale,
        }

    def ground_single(
        self,
        image: Union[str, Image.Image],
        phrase: str,
        **kwargs,
    ) -> dict:
        """单实例短语定位。"""
        img = self._load(image)
        img_small, scale, (orig_w, orig_h) = self._smart_resize(img)

        t0 = time.time()
        raw = super().ground_single(img_small, phrase, **{**self.config.to_detect_dict(), **kwargs})
        infer_ms = (time.time() - t0) * 1000

        boxes = self.parse_boxes_scaled(raw["answer"], orig_w, orig_h)
        _log.info("[tool] ground_single | %.0fms | %d boxes | %dx%d→%dx%d",
                  infer_ms, len(boxes), orig_w, orig_h, *img_small.size)

        return {
            "answer": raw["answer"],
            "boxes": boxes,
            "original_size": (orig_w, orig_h),
            "resized_size": img_small.size,
            "scale_factor": scale,
        }

    def detect_text(
        self,
        image: Union[str, Image.Image],
        **kwargs,
    ) -> dict:
        """文字检测。"""
        img = self._load(image)
        img_small, scale, (orig_w, orig_h) = self._smart_resize(img)

        t0 = time.time()
        raw = super().detect_text(img_small, **{**self.config.to_detect_dict(), **kwargs})
        infer_ms = (time.time() - t0) * 1000

        boxes = self.parse_boxes_scaled(raw["answer"], orig_w, orig_h)
        _log.info("[tool] detect_text | %.0fms | %d boxes | %dx%d→%dx%d",
                  infer_ms, len(boxes), orig_w, orig_h, *img_small.size)

        return {
            "answer": raw["answer"],
            "boxes": boxes,
            "original_size": (orig_w, orig_h),
            "resized_size": img_small.size,
            "scale_factor": scale,
        }

    def ground_gui(
        self,
        image: Union[str, Image.Image],
        phrase: str,
        output_type: str = "box",
        **kwargs,
    ) -> dict:
        """GUI 元素定位。"""
        img = self._load(image)
        img_small, scale, (orig_w, orig_h) = self._smart_resize(img)

        t0 = time.time()
        raw = super().ground_gui(img_small, phrase, output_type=output_type, **{**self.config.to_detect_dict(), **kwargs})
        infer_ms = (time.time() - t0) * 1000

        result = {
            "answer": raw["answer"],
            "original_size": (orig_w, orig_h),
            "resized_size": img_small.size,
            "scale_factor": scale,
        }

        if output_type == "point":
            result["points"] = self.parse_points_scaled(raw["answer"], orig_w, orig_h)
            items = len(result["points"])
        else:
            result["boxes"] = self.parse_boxes_scaled(raw["answer"], orig_w, orig_h)
            items = len(result["boxes"])

        _log.info("[tool] ground_gui | %.0fms | %d items | %dx%d→%dx%d",
                  infer_ms, items, orig_w, orig_h, *img_small.size)

        return result

    def point(
        self,
        image: Union[str, Image.Image],
        phrase: str,
        **kwargs,
    ) -> dict:
        """指向定位。"""
        img = self._load(image)
        img_small, scale, (orig_w, orig_h) = self._smart_resize(img)

        t0 = time.time()
        raw = super().point(img_small, phrase, **{**self.config.to_detect_dict(), **kwargs})
        infer_ms = (time.time() - t0) * 1000

        points = self.parse_points_scaled(raw["answer"], orig_w, orig_h)
        _log.info("[tool] point | %.0fms | %d points | %dx%d→%dx%d",
                  infer_ms, len(points), orig_w, orig_h, *img_small.size)

        return {
            "answer": raw["answer"],
            "points": points,
            "original_size": (orig_w, orig_h),
            "resized_size": img_small.size,
            "scale_factor": scale,
        }

    # ── 标注 ──────────────────────────────────────────────

    def annotate(
        self,
        image: Union[str, Image.Image],
        result: dict,
        *,
        categories: Optional[list[str]] = None,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        colors: Optional[list[tuple[int, int, int]]] = None,
    ) -> Image.Image:
        """在原始图像上绘制检测/定位结果。

        Args:
            image: 原始图像（路径或 PIL.Image）。
            result: detect/ground 等方法的返回结果。
            categories: 类别名列表（用于标签）。
            line_width: 框线宽度（None 则根据图像大小自动计算）。
            font_size: 字体大小。
            colors: 颜色列表。

        Returns:
            标注后的 PIL.Image（原图分辨率）。
        """
        img = self._load(image).copy()
        draw = ImageDraw.Draw(img)
        orig_w, orig_h = img.size

        # 自动线宽
        if line_width is None:
            line_width = max(2, min(orig_w, orig_h) // 500)
        if font_size is None:
            font_size = max(12, min(orig_w, orig_h) // 40)
        if colors is None:
            colors = _DEFAULT_COLORS

        # 尝试加载字体
        font = _get_font(font_size)

        items = result.get("boxes", []) or result.get("points", [])
        for i, item in enumerate(items):
            color = colors[i % len(colors)]

            if "x1" in item:  # bounding box
                x1, y1, x2, y2 = (
                    int(item["x1"]), int(item["y1"]),
                    int(item["x2"]), int(item["y2"]),
                )
                # 安全规范化：确保 x1<=x2, y1<=y2
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                if x1 == x2: x2 += 1
                if y1 == y2: y2 += 1
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)

                # 标签
                if categories and i < len(categories):
                    label = categories[i]
                    _draw_label(draw, x1, y1, label, color, font, font_size)

            elif "x" in item:  # point
                x, y = int(item["x"]), int(item["y"])
                r = max(6, line_width * 2)
                draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=line_width)
                # 十字线
                cross = r + 4
                draw.line([x - cross, y, x + cross, y], fill=color, width=max(1, line_width // 2))
                draw.line([x, y - cross, x, y + cross], fill=color, width=max(1, line_width // 2))

        return img

    # ── 一站式方法 ────────────────────────────────────────

    def detect_and_annotate(
        self,
        image: Union[str, Image.Image],
        categories: list[str],
        **kwargs,
    ) -> tuple[dict, Image.Image]:
        """检测 + 标注，一步完成。

        Returns:
            (result_dict, annotated_image)
        """
        result = self.detect(image, categories, **kwargs)
        annotated = self.annotate(image, result, categories=categories)
        return result, annotated

    # ── helper ────────────────────────────────────────────

    @staticmethod
    def _load(image: Union[str, Image.Image]) -> Image.Image:
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        return image


# ── 字体辅助 ─────────────────────────────────────────────

def _get_font(size: int) -> Optional[ImageFont.FreeTypeFont]:
    """尝试加载字体，失败则用默认。"""
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except (OSError, IOError):
        pass
    # 尝试 Windows 中文字体
    for font_path in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]:
        try:
            return ImageFont.truetype(font_path, size=size)
        except (OSError, IOError):
            continue
    return None


def _draw_label(
    draw: ImageDraw.Draw,
    x: int,
    y: int,
    text: str,
    color: tuple,
    font,
    font_size: int,
):
    """在框上方绘制标签。"""
    bbox = draw.textbbox((x, y), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    label_y = y - th - 4 if y - th - 4 > 0 else y + 2

    draw.rectangle(
        [x, label_y, x + tw + 6, label_y + th + 4],
        fill=color,
    )
    draw.text((x + 3, label_y + 2), text, fill=(255, 255, 255), font=font)
