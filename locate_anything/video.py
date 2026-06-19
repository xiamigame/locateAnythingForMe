"""
LocateAnythingForMe — 处理层：帧源 + 流式检测

提供:
- 三种帧源生成器: iter_video / iter_camera / iter_screen
- VideoLocator: 统一的流式检测（stream），不包含任何渲染代码
"""

import logging
import os
import time
from typing import Callable, Iterator, Optional

from PIL import Image

from .tool import LocateAnythingForMe

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
    """基于 LocateAnythingForMe 的视频/摄像头/屏幕流式检测器（纯处理，无渲染）。

    Usage:
        la = LocateAnythingForMe(max_edge=1024)
        vl = VideoLocator(la, frame_interval=2)

        # 视频文件目标检测
        for result in vl.stream("video.mp4", mode="detect", categories=["person", "car"]):
            print(result["boxes"])

        # 摄像头短语定位
        for result in vl.stream(0, mode="ground", phrase="red bag"):
            print(result["boxes"])
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

    # ── 统一流式检测 ──────────────────────────────────

    def stream(
        self,
        source,
        mode: str = "detect",
        *,
        categories: Optional[list[str]] = None,
        phrase: Optional[str] = None,
        output_type: str = "box",
        frame_interval: Optional[int] = None,
        max_frames: Optional[int] = None,
        **kwargs,
    ) -> Iterator[dict]:
        """统一的流式检测入口。

        Args:
            source: 帧源（int=摄像头, "screen:N"=屏幕, 视频路径, PIL Image）。
            mode: 检测模式 — "detect" | "ground" | "text" | "gui" | "point"
            categories: detect 模式的目标类别列表。
            phrase: ground/gui/point 模式的描述文本。
            output_type: gui 模式输出类型（"box" 或 "point"）。
            frame_interval: 帧间隔（None 用默认值）。
            max_frames: 最大处理帧数。
            **kwargs: 透传给模型方法。

        Yields:
            dict: {frame_index, timestamp, image, answer, boxes/points, original_size, ...}
        """
        _log.info("stream start: mode=%s source=%s interval=%s max_frames=%s",
                  mode, source, frame_interval or self.frame_interval, max_frames)
        detect_fn = self._get_detect_fn(mode, categories, phrase, output_type, **kwargs)
        t_last = time.time()

        for idx, ts, frame in self._iter_source(source, frame_interval, max_frames):
            t0 = time.time()
            result = detect_fn(frame)
            detect_ms = (time.time() - t0) * 1000
            result["frame_index"] = idx
            result["timestamp"] = ts
            result["image"] = frame

            elapsed_s = ts - t_last if t_last else 0
            _log.info("[proc] detect #%06d | %.0fms | boxes=%d | interval=%.1fs",
                       idx, detect_ms, len(result.get("boxes", []) or result.get("points", [])), elapsed_s)
            t_last = ts
            yield result

    def _get_detect_fn(
        self,
        mode: str,
        categories: Optional[list[str]],
        phrase: Optional[str],
        output_type: str,
        **kwargs,
    ) -> Callable[[Image.Image], dict]:
        """根据 mode 构建检测函数。"""
        if mode == "detect":
            if not categories:
                raise ValueError("detect 模式需要 categories")
            return lambda frame: self.model.detect(frame, categories, **kwargs)

        if mode == "ground":
            if not phrase:
                raise ValueError("ground 模式需要 phrase")
            return lambda frame: self.model.ground_multi(frame, phrase, **kwargs)

        if mode == "text":
            return lambda frame: self.model.detect_text(frame, **kwargs)

        if mode == "gui":
            if not phrase:
                raise ValueError("gui 模式需要 phrase")
            return lambda frame: self.model.ground_gui(frame, phrase, output_type=output_type, **kwargs)

        if mode == "point":
            if not phrase:
                raise ValueError("point 模式需要 phrase")
            return lambda frame: self.model.point(frame, phrase, **kwargs)

        raise ValueError(f"未知检测模式: {mode}，支持 detect/ground/text/gui/point")
