"""
LocateAnythingForMe — 视频 / 摄像头 / 屏幕 实时检测模块

基于 LocateAnythingForMe 单图检测能力，封装成流式视频检测：
- 帧源生成器（iter_video / iter_camera / iter_screen）
- VideoLocator 类（组合 LocateAnythingForMe，提供 detect_stream 等流式方法）
- @frame_processor 装饰器（包装单帧处理函数，自动适配多种输入源）
"""

import functools
import logging
import os
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Optional, Union, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .tool import LocateAnythingForMe, _DEFAULT_COLORS

_log = logging.getLogger("video")

# ── 视频文件后缀 ─────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}

# ── 帧源生成器 ───────────────────────────────────────────

def iter_video(
    video_path: str,
    frame_interval: int = 2,
    max_frames: Optional[int] = None,
) -> Iterator[tuple[int, float, Image.Image]]:
    """从视频文件按间隔抽取帧。

    Args:
        video_path: 视频文件路径。
        frame_interval: 每隔 N 帧取 1 帧（1=全部，2=每 2 帧取 1）。
        max_frames: 最多返回多少帧（None 不限）。

    Yields:
        (frame_index, timestamp_seconds, pil_image)
    """
    import decord
    vr = decord.VideoReader(video_path)
    fps = float(vr.get_avg_fps() or 30.0)
    total = len(vr)

    count = 0
    for i in range(0, total, frame_interval):
        if max_frames is not None and count >= max_frames:
            break
        try:
            frame_arr = vr[i].asnumpy()
            ts = i / fps
            yield i, ts, Image.fromarray(frame_arr)
            count += 1
        except Exception:
            # 个别帧损坏则跳过
            continue


def iter_camera(
    camera_id: int = 0,
    frame_interval: int = 2,
    max_frames: Optional[int] = None,
) -> Iterator[tuple[int, float, Image.Image]]:
    """从摄像头实时捕获帧。

    Args:
        camera_id: cv2 摄像头编号（0 为默认）。
        frame_interval: 每隔 N 帧取 1 帧。
        max_frames: 最多返回多少帧（None 不限，Ctrl+C 退出）。

    Yields:
        (frame_index, timestamp_seconds, pil_image)
    """
    import cv2
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 {camera_id}")

    base_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    t0 = time.time()
    i = 0
    count = 0

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if i % frame_interval == 0:
                if max_frames is not None and count >= max_frames:
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                ts = time.time() - t0
                yield i, ts, Image.fromarray(frame_rgb)
                count += 1
            i += 1
    finally:
        cap.release()


def iter_screen(
    monitor: int = 0,
    frame_interval: int = 2,
    max_frames: Optional[int] = None,
    region: Optional[tuple[int, int, int, int]] = None,
) -> Iterator[tuple[int, float, Image.Image]]:
    """从屏幕实时捕获帧。

    Args:
        monitor: 显示器编号。
                 0 = 全虚拟桌面（所有显示器聚合），
                 1 = 主显示器，2 = 第二个显示器，以此类推。
        frame_interval: 每隔 N 帧取 1 帧。
        max_frames: 最多返回多少帧（None 不限，Ctrl+C 退出）。
        region: 自定义区域 (left, top, width, height)，指定后忽略 monitor 参数。

    Yields:
        (frame_index, timestamp_seconds, pil_image)
    """
    import mss as _mss
    sct = _mss.mss()

    # 自定义区域优先
    if region is not None:
        monitor_geo = {"left": region[0], "top": region[1],
                       "width": region[2], "height": region[3]}
    else:
        # mss.monitors[0] = 全虚拟桌面, [1] = 主显示器, [2] = 第二显示器...
        if monitor >= len(sct.monitors):
            raise RuntimeError(
                f"显示器 {monitor} 不存在（共 {len(sct.monitors)} 个，编号 0~{len(sct.monitors) - 1}）"
            )
        monitor_geo = sct.monitors[monitor]

    i = 0
    count = 0
    t0 = time.time()

    try:
        while True:
            if i % frame_interval == 0:
                if max_frames is not None and count >= max_frames:
                    break
                sct_img = sct.grab(monitor_geo)
                frame = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
                ts = time.time() - t0
                yield i, ts, frame
                count += 1
            i += 1
    finally:
        sct.close()


# ── 帧源工厂 ────────────────────────────────────────────

def _resolve_source(
    source,
    frame_interval: int = 2,
    max_frames: Optional[int] = None,
):
    """根据输入类型自动选择合适的帧源。

    Returns:
        (Iterator[tuple[int, float, Image.Image]], is_image: bool)
        如果是单张图片则 is_image=True，返回单帧迭代器。
    """
    # 摄像头
    if isinstance(source, int):
        return iter_camera(source, frame_interval=frame_interval, max_frames=max_frames), False

    if isinstance(source, str):
        # screen:N
        if source.startswith("screen:"):
            monitor = int(source.split(":", 1)[1])
            return iter_screen(monitor, frame_interval=frame_interval, max_frames=max_frames), False
        # camera:N
        if source.startswith("camera:"):
            cam_id = int(source.split(":", 1)[1])
            return iter_camera(cam_id, frame_interval=frame_interval, max_frames=max_frames), False
        # 视频文件
        ext = os.path.splitext(source)[1].lower()
        if ext in VIDEO_EXTS:
            return iter_video(source, frame_interval=frame_interval, max_frames=max_frames), False
        # 单张图片
        img = Image.open(source).convert("RGB")
        return _single_image_iter(img), True

    # PIL Image 直接
    if isinstance(source, Image.Image):
        return _single_image_iter(source), True

    raise TypeError(f"不支持的输入源类型: {type(source)}")


def _single_image_iter(img: Image.Image) -> Iterator[tuple[int, float, Image.Image]]:
    """单张图片退化为单帧迭代器。"""
    yield 0, 0.0, img


# ── VideoLocator ─────────────────────────────────────────

class VideoLocator:
    """基于 LocateAnythingForMe 的视频/摄像头/屏幕实时目标定位器。

    使用示例:
        la = LocateAnythingForMe(max_edge=1024)
        vl = VideoLocator(la, frame_interval=2)

        # 视频文件
        for idx, ts, result in vl.detect_stream("video.mp4", ["person", "car"]):
            vl.annotate_and_save(idx, ts, result, "output/")

        # 摄像头
        for idx, ts, result in vl.detect_stream(0, ["face"]):
            vl.annotate_and_save(idx, ts, result, "output/")

        # 屏幕
        for idx, ts, result in vl.detect_stream("screen:1", ["button"]):
            vl.annotate_and_save(idx, ts, result, "output/")
    """

    def __init__(
        self,
        model: LocateAnythingForMe,
        frame_interval: int = 2,
    ):
        """
        Args:
            model: LocateAnythingForMe 实例。
            frame_interval: 默认帧间隔（1=不跳帧，2=每 2 帧取 1）。
        """
        self.model = model
        self.frame_interval = frame_interval

    # ── 内部辅助 ──────────────────────────────────────

    def _iter_source(self, source, frame_interval=None, max_frames=None):
        """统一帧源：根据 source 类型路由，按 interval 跳帧。"""
        interval = frame_interval if frame_interval is not None else self.frame_interval
        frames_iter, _ = _resolve_source(source, frame_interval=interval, max_frames=max_frames)
        return frames_iter

    # ── 流式检测 ──────────────────────────────────────

    def detect_stream(
        self,
        source,
        categories: list[str],
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[dict]:
        """视频目标检测流。

        Yields:
            dict: {frame_index, timestamp, image, answer, boxes, original_size, ...}
        """
        for idx, ts, frame in self._iter_source(source, frame_interval, max_frames):
            result = self.model.detect(frame, categories, **kwargs)
            result["frame_index"] = idx
            result["timestamp"] = ts
            result["image"] = frame
            yield result

    def ground_multi_stream(
        self,
        source,
        phrase: str,
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[dict]:
        """视频多实例短语定位流。"""
        for idx, ts, frame in self._iter_source(source, frame_interval, max_frames):
            result = self.model.ground_multi(frame, phrase, **kwargs)
            result["frame_index"] = idx
            result["timestamp"] = ts
            result["image"] = frame
            yield result

    def ground_single_stream(
        self,
        source,
        phrase: str,
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[dict]:
        """视频单实例短语定位流。"""
        for idx, ts, frame in self._iter_source(source, frame_interval, max_frames):
            result = self.model.ground_single(frame, phrase, **kwargs)
            result["frame_index"] = idx
            result["timestamp"] = ts
            result["image"] = frame
            yield result

    def detect_text_stream(
        self,
        source,
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[dict]:
        """视频文字检测流。"""
        for idx, ts, frame in self._iter_source(source, frame_interval, max_frames):
            result = self.model.detect_text(frame, **kwargs)
            result["frame_index"] = idx
            result["timestamp"] = ts
            result["image"] = frame
            yield result

    def ground_gui_stream(
        self,
        source,
        phrase: str,
        output_type: str = "box",
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[dict]:
        """视频 GUI 元素定位流。"""
        for idx, ts, frame in self._iter_source(source, frame_interval, max_frames):
            result = self.model.ground_gui(frame, phrase, output_type=output_type, **kwargs)
            result["frame_index"] = idx
            result["timestamp"] = ts
            result["image"] = frame
            yield result

    def point_stream(
        self,
        source,
        phrase: str,
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[dict]:
        """视频指向定位流。"""
        for idx, ts, frame in self._iter_source(source, frame_interval, max_frames):
            result = self.model.point(frame, phrase, **kwargs)
            result["frame_index"] = idx
            result["timestamp"] = ts
            result["image"] = frame
            yield result

    # ── 标注 ──────────────────────────────────────────

    def annotate(
        self,
        image: Image.Image,
        result: dict,
        *,
        categories: Optional[list[str]] = None,
        line_width: Optional[int] = None,
        font_size: Optional[int] = None,
        colors: Optional[list[tuple[int, int, int]]] = None,
    ) -> Image.Image:
        """在图像上绘制检测结果（委托给模型）。"""
        return self.model.annotate(
            image, result,
            categories=categories,
            line_width=line_width,
            font_size=font_size,
            colors=colors,
        )

    def annotate_frame(
        self,
        result: dict,
        categories: Optional[list[str]] = None,
        **kwargs,
    ) -> Image.Image:
        """对单帧检测结果进行标注，并在角落绘制帧信息。

        Returns:
            标注后的 PIL.Image（已包含帧号 + 时间戳水印）。
        """
        img = result["image"].copy()
        draw = ImageDraw.Draw(img)

        # 帧信息水印
        idx = result.get("frame_index", 0)
        ts = result.get("timestamp", 0.0)
        watermark = f"#{idx}  {ts:.1f}s"

        # 自动字号
        fsize = kwargs.get("font_size", max(12, min(img.size) // 40))
        font = _get_font(fsize)
        bbox = draw.textbbox((0, 0), watermark, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # 左上角半透明背景 + 文字
        draw.rectangle([0, 0, tw + 12, th + 8], fill=(0, 0, 0, 128))
        draw.text((6, 4), watermark, fill=(255, 255, 255), font=font)

        # 检测标注
        img = self.model.annotate(img, result, categories=categories, **kwargs)
        return img

    # ── 异步流（显示线程全速输出，检测线程后台运行）───

    def _async_stream(
        self,
        source,
        detect_fn,
        annotate_kwargs,
        detect_interval: int = 2,
        max_frames: Optional[int] = None,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """异步检测流 —— 单一帧源，显示/检测分离。非检测帧轻量叠加缓存结果。

        Yields:
            (frame_index, timestamp, annotated_image, result_dict_or_None)
        """
        _LOCK = threading.Lock()
        _cache: dict = {"result": None, "overlay": None}  # overlay = 透明标注层
        _work_queue = queue.Queue(maxsize=1)

        def _detection_worker():
            while True:
                item = _work_queue.get()
                if item is None:
                    break
                _idx, _ts, _frame = item
                try:
                    result = detect_fn(_frame)
                    result["frame_index"] = _idx
                    result["timestamp"] = _ts
                    result["image"] = _frame
                    # 预渲染标注到独立透明层
                    overlay = self._render_overlay(result, **annotate_kwargs)
                    with _LOCK:
                        _cache["result"] = result
                        _cache["overlay"] = overlay
                except Exception as e:
                    print(f"[async] 检测帧 #{_idx} 失败: {e}", flush=True)

        worker = threading.Thread(target=_detection_worker, daemon=True)
        worker.start()

        count = 0
        wm_font = _get_font(16)

        for idx, ts, frame in self._iter_source(source, frame_interval=1, max_frames=max_frames):
            count += 1
            if max_frames is not None and count > max_frames:
                break

            if idx % detect_interval == 0:
                try:
                    _work_queue.put_nowait((idx, ts, frame.copy()))
                except queue.Full:
                    pass

            with _LOCK:
                overlay = _cache["overlay"]
                cached_result = _cache["result"]

            if overlay is not None:
                out = frame.copy()
                out.paste(overlay, (0, 0), overlay)
                self._draw_watermark_fast(out, idx, ts, wm_font)
                display = dict(cached_result) if cached_result else {}
                display["image"] = frame
                display["frame_index"] = idx
                display["timestamp"] = ts
                yield idx, ts, out, display if cached_result else None
            else:
                out = frame.copy()
                self._draw_watermark_fast(out, idx, ts, wm_font)
                yield idx, ts, out, None

        _work_queue.put(None)
        worker.join(timeout=5)

    def _render_overlay(self, result: dict, **kwargs) -> Image.Image:
        """预渲染标注到独立 RGBA 透明图层，供后续帧复用。"""
        img = result["image"]
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        line_width = kwargs.pop("line_width", max(2, min(img.size) // 500))
        font_size = kwargs.pop("font_size", max(12, min(img.size) // 40))
        colors = kwargs.pop("colors", _DEFAULT_COLORS)
        categories = kwargs.pop("categories", None)
        font = _get_font(font_size)

        items = result.get("boxes", []) or result.get("points", [])
        for i, item in enumerate(items):
            color = colors[i % len(colors)]
            if "x1" in item:
                x1, y1, x2, y2 = int(item["x1"]), int(item["y1"]), int(item["x2"]), int(item["y2"])
                if x1 > x2: x1, x2 = x2, x1
                if y1 > y2: y1, y2 = y2, y1
                draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
                if categories and i < len(categories):
                    # 绘制标签
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


    @staticmethod
    def _draw_watermark_fast(img: Image.Image, idx: int, ts: float, font: ImageFont.FreeTypeFont):
        """在图像左上角快速绘制帧号水印。"""
        draw = ImageDraw.Draw(img)
        text = f"#{idx}  {ts:.1f}s"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle([0, 0, tw + 10, th + 6], fill=(0, 0, 0, 180))
        draw.text((5, 3), text, fill=(255, 255, 255), font=font)

    # ── 平滑流（检测跳帧，展示不跳帧）───────────────

    def detect_and_annotate_smooth(
        self,
        source,
        categories: list[str],
        detect_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """目标检测异步流：原始帧率展示，后台间隔检测。"""
        interval = detect_interval if detect_interval is not None else self.frame_interval
        return self._async_stream(
            source,
            lambda f: self.model.detect(f, categories, **kwargs),
            {"categories": categories, **kwargs},
            detect_interval=interval,
            max_frames=max_frames,
        )

    def ground_multi_and_annotate_smooth(
        self,
        source,
        phrase: str,
        detect_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """多实例定位异步流。"""
        interval = detect_interval if detect_interval is not None else self.frame_interval
        return self._async_stream(
            source,
            lambda f: self.model.ground_multi(f, phrase, **kwargs),
            {},
            detect_interval=interval,
            max_frames=max_frames,
        )

    def ground_single_and_annotate_smooth(
        self,
        source,
        phrase: str,
        detect_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """单实例定位异步流。"""
        interval = detect_interval if detect_interval is not None else self.frame_interval
        return self._async_stream(
            source,
            lambda f: self.model.ground_single(f, phrase, **kwargs),
            {},
            detect_interval=interval,
            max_frames=max_frames,
        )

    def detect_text_and_annotate_smooth(
        self,
        source,
        detect_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """文字检测异步流。"""
        interval = detect_interval if detect_interval is not None else self.frame_interval
        return self._async_stream(
            source,
            lambda f: self.model.detect_text(f, **kwargs),
            {},
            detect_interval=interval,
            max_frames=max_frames,
        )

    def ground_gui_and_annotate_smooth(
        self,
        source,
        phrase: str,
        output_type: str = "box",
        detect_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """GUI 定位异步流。"""
        interval = detect_interval if detect_interval is not None else self.frame_interval
        return self._async_stream(
            source,
            lambda f: self.model.ground_gui(f, phrase, output_type=output_type, **kwargs),
            {},
            detect_interval=interval,
            max_frames=max_frames,
        )

    def point_and_annotate_smooth(
        self,
        source,
        phrase: str,
        detect_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """指向定位异步流。"""
        interval = detect_interval if detect_interval is not None else self.frame_interval
        return self._async_stream(
            source,
            lambda f: self.model.point(f, phrase, **kwargs),
            {},
            detect_interval=interval,
            max_frames=max_frames,
        )

    # ── 注解流 ────────────────────────────────────────

    def detect_and_annotate_stream(
        self,
        source,
        categories: list[str],
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """检测 + 标注流（仅检测帧，跳帧输出）。

        Yields:
            (frame_index, timestamp, annotated_image, result_dict)
        """
        for result in self.detect_stream(source, categories, frame_interval=frame_interval,
                                          max_frames=max_frames, **kwargs):
            annotated = self.annotate_frame(result, categories=categories, **kwargs)
            yield result["frame_index"], result["timestamp"], annotated, result

    def ground_multi_and_annotate_stream(
        self,
        source,
        phrase: str,
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[tuple[int, float, Image.Image, dict]]:
        """多实例定位 + 标注流。"""
        for result in self.ground_multi_stream(source, phrase, frame_interval=frame_interval,
                                                max_frames=max_frames, **kwargs):
            annotated = self.annotate_frame(result, **kwargs)
            yield result["frame_index"], result["timestamp"], annotated, result

    # ── 保存 ──────────────────────────────────────────

    @staticmethod
    def save_frame(
        image: Image.Image,
        frame_index: int,
        output_dir: str,
        prefix: str = "frame",
    ):
        """保存单帧到指定目录。"""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{prefix}_{frame_index:06d}.jpg")
        image.save(path)
        return path


# ── @frame_processor 装饰器 ──────────────────────────────

def frame_processor(
    frame_interval: int = 2,
    max_frames: Optional[int] = None,
):
    """装饰器：将单帧处理函数包装成能处理视频/摄像头/屏幕/图片的流式函数。

    被装饰函数签名:
        func(la: LocateAnythingForMe, frame: Image.Image, **kwargs) -> Image.Image

    包装后函数:
        wrapper(source, la=None, **kwargs) -> Iterator[tuple[int, float, Image.Image]]

    Args:
        frame_interval: 帧间隔。
        max_frames: 最大帧数。

    Example:
        @frame_processor(frame_interval=2, max_frames=16)
        def detect_person(la, frame):
            result = la.detect(frame, ["person"])
            return la.annotate(frame, result, categories=["person"])

        for idx, ts, annotated in detect_person("video.mp4", la=la):
            annotated.save(f"out/frame_{idx:04d}.jpg")
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(
            source=None,
            *,
            la: Optional[LocateAnythingForMe] = None,
            camera: Optional[int] = None,
            screen: Optional[int] = None,
            frame_interval_override: Optional[int] = None,
            max_frames_override: Optional[int] = None,
            **kwargs,
        ) -> Iterator[tuple[int, float, Image.Image]]:
            interval = frame_interval_override or frame_interval
            maxf = max_frames_override or max_frames

            # 确定帧源
            if camera is not None:
                frames_iter = iter_camera(camera, frame_interval=interval, max_frames=maxf)
            elif screen is not None:
                frames_iter = iter_screen(screen, frame_interval=interval, max_frames=maxf)
            elif source is not None:
                frames_iter, is_img = _resolve_source(source, frame_interval=interval, max_frames=maxf)
            else:
                raise ValueError("必须提供 source, camera 或 screen 之一")

            for idx, ts, frame in frames_iter:
                try:
                    result = func(la, frame, **kwargs)
                    yield idx, ts, result
                except Exception as e:
                    # 单帧失败不终止整个流
                    print(f"[frame_processor] 帧 #{idx} 处理失败: {e}")
                    yield idx, ts, frame  # 返回原帧

        # 把被装饰函数引用挂到 wrapper 上，方便测试
        wrapper.__wrapped__ = func
        return wrapper
    return decorator


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
