"""
LocateAnythingForMe — 渲染层

标注绘制、水印、叠加层合成、异步平滑渲染。
与检测模式（detect/ground/text/gui/point）无关，只接受检测结果并渲染。

提供:
- 模块级工具: draw_watermark, render_overlay, annotate_frame
- VideoRenderer: 包装 VideoLocator，提供标注流 + 异步平滑流
"""

import logging
import os as _os
import queue
import threading
import time
from contextlib import redirect_stdout as _redirect_stdout
from typing import Callable, Iterator, Optional

from PIL import Image, ImageDraw, ImageFont

from .tool import _DEFAULT_COLORS

_log = logging.getLogger("rendering")

# ── 字体辅助 ─────────────────────────────────────────────

_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """加载字体（带缓存），失败则用默认。"""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    font = None
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]:
        try:
            font = ImageFont.truetype(font_path, size=size)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


# ── 标注工具 ─────────────────────────────────────────────


def draw_watermark(
    img: Image.Image,
    idx: int,
    ts: float,
    font: Optional[ImageFont.FreeTypeFont] = None,
) -> None:
    """在图像左上角绘制帧号和时间戳水印（原地修改）。"""
    if font is None:
        font = _get_font(16)
    draw = ImageDraw.Draw(img)
    text = f"#{idx}  {ts:.1f}s"
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.rectangle([0, 0, tw + 10, th + 6], fill=(0, 0, 0, 180))
    draw.text((5, 3), text, fill=(255, 255, 255), font=font)


def render_overlay(result: dict, **kwargs) -> Image.Image:
    """创建 RGBA 透明标注叠加层（框/点/标签），用于异步平滑流复用。

    Args:
        result: 检测结果 dict，需包含 "image" 键。
        **kwargs: line_width, font_size, colors, categories 等。

    Returns:
        RGBA 透明图层 PIL.Image，非标注区域全透明。
    """
    img = result["image"]
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    line_width = kwargs.get("line_width", max(2, min(img.size) // 500))
    font_size = kwargs.get("font_size", max(12, min(img.size) // 40))
    colors = kwargs.get("colors", _DEFAULT_COLORS)
    categories = kwargs.get("categories", None)
    font = _get_font(font_size)

    items = result.get("boxes", []) or result.get("points", [])
    for i, item in enumerate(items):
        color = colors[i % len(colors)]
        if "x1" in item:
            x1, y1, x2, y2 = int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
            if categories and i < len(categories):
                label = categories[i]
                tbox = draw.textbbox((x1, y1), label, font=font)
                tw, th = tbox[2] - tbox[0], tbox[3] - tbox[1]
                ly = y1 - th - 4 if y1 - th - 4 > 0 else y1 + 2
                draw.rectangle([x1, ly, x1 + tw + 6, ly + th + 4], fill=color)
                draw.text((x1 + 3, ly + 2), label, fill=(255, 255, 255), font=font)
        elif "x" in item:
            x, y = int(item["x"]), int(item["y"])
            r = max(6, line_width * 2)
            draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=line_width)
            cross = r + 4
            draw.line([x - cross, y, x + cross, y], fill=color, width=max(1, line_width // 2))
            draw.line([x, y - cross, x, y + cross], fill=color, width=max(1, line_width // 2))

    return overlay


def annotate_frame(
    img: Image.Image,
    result: dict,
    model,
    **kwargs,
) -> Image.Image:
    """单帧标注：水印 + model.annotate() 检测框/点。

    Args:
        img: 原始帧。
        result: 检测结果 dict。
        model: LocateAnythingForMe 实例（用于 annotate）。
        **kwargs: 透传给 model.annotate()（categories, line_width, ...）。

    Returns:
        标注后的 PIL.Image。
    """
    annotated = img.copy()
    idx = result.get("frame_index", 0)
    ts = result.get("timestamp", 0.0)
    draw_watermark(annotated, idx, ts)
    t0 = time.time()
    out = model.annotate(annotated, result, **kwargs)
    _log.info("[render] annotate_frame #%06d | %.0fms", idx, (time.time() - t0) * 1000)
    return out


# ── VideoRenderer ────────────────────────────────────────


class VideoRenderer:
    """渲染层：包装 VideoLocator，提供标注 + 异步平滑渲染。

    不与检测模式耦合——调用方提供检测函数和标注参数。

    用法:
        vl = VideoLocator(model, frame_interval=2)
        vr = VideoRenderer(vl)

        # 同步：检测 → 标注
        stream = vl.stream(source, mode="detect", categories=["person"])
        for idx, ts, img, result in vr.annotate_stream(stream, categories=["person"]):
            img.save(f"frame_{idx:06d}.jpg")

        # 异步平滑：后台间隔检测，全帧率渲染
        for idx, ts, img, result in vr.annotate_smooth(
            camera_id,
            detect_fn=lambda f: model.detect(f, ["person"]),
            detect_interval=4, categories=["person"],
        ):
            ...
    """

    def __init__(self, video_locator):
        self.vl = video_locator
        self.model = video_locator.model

    # ── 同步标注流 ──────────────────────────────────

    def annotate_stream(
        self,
        detect_iter: Iterator[dict],
        **annotate_kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """对检测流逐帧标注 + 水印。

        Args:
            detect_iter: VideoLocator.stream() 返回的检测结果迭代器。
            **annotate_kwargs: 透传给 model.annotate()（categories, line_width, ...）。

        Yields:
            (frame_index, timestamp, annotated_image, result_dict)
        """
        for result in detect_iter:
            annotated = annotate_frame(
                result["image"], result, self.model, **annotate_kwargs
            )
            yield result["frame_index"], result["timestamp"], annotated, result

    # ── 异步平滑流 ──────────────────────────────────

    def annotate_smooth(
        self,
        source,
        detect_fn: Callable[[Image.Image], dict],
        *,
        detect_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **annotate_kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """异步平滑流：后台间隔检测 + 全帧率渲染叠加层。

        检测线程按 detect_interval 跳帧运行模型推理，
        主线程以原生帧率（frame_interval=1）输出所有帧，
        非检测帧复用上一次的检测叠加层 + 水印。

        Args:
            source: 帧源（同 VideoLocator.stream()）。
            detect_fn: 检测函数，签名为 fn(frame: Image.Image) -> dict。
            detect_interval: 每 N 帧检测一次（默认用 VideoLocator.frame_interval）。
            max_frames: 最大输出帧数。
            **annotate_kwargs: 透传给 render_overlay()（line_width, colors, categories, ...）。

        Yields:
            (frame_index, timestamp, annotated_image, result_dict_or_None)
            非检测帧的 result 为 None。
        """
        interval = detect_interval if detect_interval is not None else self.vl.frame_interval
        return self._async_stream(source, detect_fn, annotate_kwargs, interval, max_frames)

    def _async_stream(
        self,
        source,
        detect_fn: Callable[[Image.Image], dict],
        annotate_kwargs: dict,
        detect_interval: int,
        max_frames: Optional[int],
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """异步检测流 —— 单一帧源，显示/检测分离。

        后台线程跑模型检测，缓存最新结果。
        主线程每帧基于**当前画面**重新渲染叠加层，避免帧回滚。
        支持 GeneratorExit 优雅停止（客户端断开时清理线程和摄像头）。
        """
        _LOCK = threading.Lock()
        _cache: dict = {"result": None}
        _work_queue = queue.Queue(maxsize=1)
        _stop = threading.Event()

        def _detection_worker():
            while not _stop.is_set():
                try:
                    item = _work_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None or _stop.is_set():
                    break
                _idx, _ts, _frame = item
                try:
                    t0 = time.time()
                    # 抑制模型内部的 print（Statistic Info 等调试输出）
                    with open(_os.devnull, "w") as _devnull, _redirect_stdout(_devnull):
                        result = detect_fn(_frame)
                    detect_ms = (time.time() - t0) * 1000
                    result["frame_index"] = _idx
                    result["timestamp"] = _ts
                    result["image"] = _frame
                    with _LOCK:
                        _cache["result"] = result
                    items = len(result.get("boxes", []) or result.get("points", []))
                    _log.info("[async] detect #%06d | %.0fms | %d items",
                              _idx, detect_ms, items)
                except Exception as e:
                    _log.error("[async] detect #%d failed: %s", _idx, e)

        worker = threading.Thread(target=_detection_worker, daemon=True)
        worker.start()

        count = 0
        t_summary = time.time()
        wm_font = _get_font(16)
        _log.info("[async] start: interval=%d max_frames=%s", detect_interval, max_frames or "∞")

        try:
            for idx, ts, frame in self.vl._iter_source(source, frame_interval=1, max_frames=max_frames):
                count += 1
                if max_frames is not None and count > max_frames:
                    break

                if idx % detect_interval == 0:
                    try:
                        _work_queue.put_nowait((idx, ts, frame.copy()))
                    except queue.Full:
                        _log.info("[async] queue full, skip #%d", idx)

                with _LOCK:
                    cached_result = _cache["result"]

                t0_render = time.time()
                if cached_result is not None:
                    # 在当前帧上渲染叠加层，避免帧回滚
                    cached_result["image"] = frame
                    overlay = render_overlay(cached_result, **annotate_kwargs)
                    out = frame.copy()
                    out.paste(overlay, (0, 0), overlay)
                    draw_watermark(out, idx, ts, wm_font)
                    render_ms = (time.time() - t0_render) * 1000
                    display = dict(cached_result)
                    display["image"] = frame
                    display["frame_index"] = idx
                    display["timestamp"] = ts
                    yield idx, ts, out, display
                else:
                    out = frame.copy()
                    draw_watermark(out, idx, ts, wm_font)
                    render_ms = (time.time() - t0_render) * 1000
                    yield idx, ts, out, None

                # 每 30 帧输出一次渲染性能摘要
                if count % 30 == 0:
                    elapsed = time.time() - t_summary
                    fps = 30 / elapsed
                    _log.info("[async] render fps=%.1f render_ms=%.0f frame=%d/%s",
                              fps, render_ms, count, max_frames or "∞")
                    t_summary = time.time()
        finally:
            _stop.set()
            # 清空待处理队列，发 poison pill，不 join（daemon 线程会自动退出）
            while True:
                try:
                    _work_queue.get_nowait()
                except queue.Empty:
                    break
            try:
                _work_queue.put_nowait(None)
            except queue.Full:
                pass
