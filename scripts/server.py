#!/usr/bin/env python
"""
LocateAnythingForMe — 自建 Web 前端 (FastAPI + MJPEG)

摄像头/屏幕用 MJPEG 流（原生帧率），视频/图片用普通 API。

用法:
    python scripts/server.py                          # http://127.0.0.1:8765
    python scripts/server.py --port 8080 --max-edge 512
"""
import argparse
import io
import logging
import os
import sys
import threading
import time
import zipfile
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

# 环境初始化
from locate_anything import VideoLocator, VideoRenderer, LocateAnythingForMe, LocateConfig

from PIL import Image
import numpy as np

# ── 日志 ──
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("server")

# ── 全局模型（懒加载）─────
_LA: LocateAnythingForMe = None
_LA_LOCK = threading.Lock()
_LA_CONFIG: LocateConfig = LocateConfig()


def get_model():
    global _LA
    if _LA is not None:
        return _LA
    with _LA_LOCK:
        if _LA is not None:
            return _LA
        log.info("Loading model %s (max_edge=%d generation=%s)...",
                 _LA_CONFIG.model_path, _LA_CONFIG.max_edge, _LA_CONFIG.generation_mode)
        _LA = LocateAnythingForMe(config=_LA_CONFIG)
        log.info("Model loaded.")
        return _LA


# ── MJPEG 流生成 ──────

def mjpeg_stream(gen):
    """将生成器转为 MJPEG HTTP 流。"""
    for frame, _info in gen:
        buf = io.BytesIO()
        frame.save(buf, format="JPEG", quality=80)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.getvalue() + b"\r\n")


def camera_stream_generator(camera_id, detect_interval, categories_str, max_edge):
    """摄像头 MJPEG 生成器。"""
    la = get_model()
    la.max_edge = int(max_edge)
    categories = [c.strip() for c in categories_str.split(",") if c.strip()] or ["person"]
    vl = VideoLocator(la, frame_interval=int(detect_interval))
    vr = VideoRenderer(vl)

    try:
        for idx, ts, annotated, result in vr.annotate_smooth(
            camera_id,
            detect_fn=lambda f: la.detect(f, categories),
            detect_interval=int(detect_interval),
            categories=categories,
        ):
            boxes = result.get("boxes", []) if result else []
            info = f"#{idx} {ts:.1f}s | {len(boxes)} obj"
            yield annotated, info
    except GeneratorExit:
        log.info("Camera stream disconnected (camera=%d)", camera_id)
    finally:
        log.info("Camera stream stopped (camera=%d)", camera_id)


def screen_stream_generator(monitor, detect_interval, categories_str, max_edge):
    """屏幕 MJPEG 生成器。"""
    la = get_model()
    la.max_edge = int(max_edge)
    categories = [c.strip() for c in categories_str.split(",") if c.strip()] or ["button", "icon", "text"]
    vl = VideoLocator(la, frame_interval=int(detect_interval))
    vr = VideoRenderer(vl)

    try:
        for idx, ts, annotated, result in vr.annotate_smooth(
            f"screen:{monitor}",
            detect_fn=lambda f: la.detect(f, categories),
            detect_interval=int(detect_interval),
            categories=categories,
        ):
            boxes = result.get("boxes", []) if result else []
            info = f"#{idx} {ts:.1f}s | {len(boxes)} obj"
            yield annotated, info
    except GeneratorExit:
        log.info("Screen stream disconnected (monitor=%d)", monitor)
    finally:
        log.info("Screen stream stopped (monitor=%d)", monitor)


# ── FastAPI ──────

from fastapi import FastAPI, Request, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="LocateAnythingForMe")


@app.get("/", response_class=HTMLResponse)
async def index():
    tpl = (_PROJECT / "scripts" / "templates" / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(tpl)


@app.get("/api/stream/camera")
async def stream_camera(
    camera: int = Query(0),
    interval: int = Query(4),
    categories: str = Query("person,face,chair"),
    max_edge: int = Query(512),
):
    gen = camera_stream_generator(camera, interval, categories, max_edge)
    return StreamingResponse(
        mjpeg_stream(gen),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/stream/screen")
async def stream_screen(
    monitor: int = Query(1),
    interval: int = Query(8),
    categories: str = Query("button,icon,text"),
    max_edge: int = Query(512),
):
    gen = screen_stream_generator(monitor, interval, categories, max_edge)
    return StreamingResponse(
        mjpeg_stream(gen),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.post("/api/detect/image")
async def detect_image(
    image: UploadFile = File(...),
    mode: str = Form("detect"),
    categories: str = Form("person,face"),
    phrase: str = Form(""),
    max_edge: int = Form(512),
):
    la = get_model()
    la.max_edge = max_edge

    img = Image.open(io.BytesIO(await image.read())).convert("RGB")
    cats = [c.strip() for c in categories.split(",") if c.strip()]

    if mode == "detect":
        result = la.detect(img, cats)
        annotated = la.annotate(img, result, categories=cats)
    elif mode == "ground":
        result = la.ground_multi(img, phrase or "person")
        annotated = la.annotate(img, result)
    elif mode == "text":
        result = la.detect_text(img)
        annotated = la.annotate(img, result)
    elif mode == "gui":
        result = la.ground_gui(img, phrase or "button")
        annotated = la.annotate(img, result)
    elif mode == "point":
        result = la.point(img, phrase or "center")
        annotated = la.annotate(img, result)
    else:
        return {"error": f"Unknown mode: {mode}"}

    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=90)
    boxes = result.get("boxes", [])
    points = result.get("points", [])
    return Response(
        content=buf.getvalue(),
        media_type="image/jpeg",
        headers={"X-Result": f"{len(boxes)} boxes, {len(points)} points"},
    )


@app.post("/api/detect/video")
async def detect_video(
    video: UploadFile = File(...),
    interval: int = Form(4),
    categories: str = Form("person,car"),
    max_edge: int = Form(512),
):
    la = get_model()
    la.max_edge = max_edge
    cats = [c.strip() for c in categories.split(",") if c.strip()] or ["person"]

    # 保存上传视频到临时文件
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(await video.read())
    tmp.close()

    vl = VideoLocator(la, frame_interval=interval)
    vr = VideoRenderer(vl)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        stream = vl.stream(tmp.name, mode="detect", categories=cats)
        for idx, ts, annotated, result in vr.annotate_stream(stream, categories=cats):
            frame_buf = io.BytesIO()
            annotated.save(frame_buf, format="JPEG", quality=85)
            zf.writestr(f"frame_{idx:06d}.jpg", frame_buf.getvalue())

    os.unlink(tmp.name)
    return Response(content=buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": "attachment; filename=detected_frames.zip"})


# ── 静态文件 ──────
@app.get("/api/model/status")
async def model_status():
    return {"loaded": _LA is not None, "config": _LA_CONFIG.to_dict()}


# ── 入口 ──────

def main():
    parser = argparse.ArgumentParser(description="LocateAnythingForMe Web Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="nvidia/LocateAnything-3B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-edge", type=int, default=512)
    parser.add_argument("--generation-mode", default="fast", choices=["fast", "slow", "hybrid"])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    global _LA_CONFIG
    _LA_CONFIG = LocateConfig(
        model_path=args.model,
        device=args.device,
        max_edge=args.max_edge,
        generation_mode=args.generation_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    # 后台预加载模型
    threading.Thread(target=get_model, daemon=True).start()

    import uvicorn
    log.info("Starting server: http://%s:%s", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
