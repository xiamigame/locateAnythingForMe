"""
locate_anything/video.py + rendering.py 纯函数单元测试（无需加载模型）
"""
import sys
import os
import tempfile
import ast
from pathlib import Path

# Setup path
_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT)

import locate_anything  # noqa: F401  环境初始化在 __init__.py 里

from PIL import Image

# ─── 测试 1: 模块可解析 ───
def test_syntax():
    """所有文件语法正确"""
    files = [
        "locate_anything/video.py",
        "locate_anything/rendering.py",
        "locate_anything/__init__.py",
        "scripts/video.py",
        "scripts/server.py",
    ]
    for f in files:
        with open(f, encoding="utf-8") as fh:
            ast.parse(fh.read())
    print("PASS test_syntax")

# ─── 测试 2: 常量 ───
def test_video_exts():
    from locate_anything.video import VIDEO_EXTS
    assert ".mp4" in VIDEO_EXTS
    assert ".avi" in VIDEO_EXTS
    assert ".mov" in VIDEO_EXTS
    assert len(VIDEO_EXTS) >= 5
    print("PASS test_video_exts")

# ─── 测试 3: _resolve_source ───
def test_resolve_source():
    from locate_anything.video import _resolve_source, _single_image_iter

    # PIL Image → is_image=True
    img = Image.new("RGB", (100, 100))
    it, is_img = _resolve_source(img)
    assert is_img is True
    idx, ts, frame = next(it)
    assert idx == 0
    assert ts == 0.0
    assert frame.size == (100, 100)

    # camera:int
    it, is_img = _resolve_source(0)
    assert is_img is False

    # camera:N string
    it, is_img = _resolve_source("camera:1")
    assert is_img is False

    # screen:N string
    it, is_img = _resolve_source("screen:2")
    assert is_img is False

    # video路径 (即使是假路径也按视频识别)
    it, is_img = _resolve_source("test.mp4")
    assert is_img is False
    it, is_img = _resolve_source("video.avi")
    assert is_img is False
    it, is_img = _resolve_source("movie.mkv")
    assert is_img is False

    # 非视频后缀的路径 → 按图片（会因文件不存在报 FileNotFoundError）
    try:
        _resolve_source("notexist.jpg")
    except FileNotFoundError:
        pass  # 预期行为

    print("PASS test_resolve_source")

# ─── 测试 4: VideoLocator 类结构 ───
def test_video_locator_structure():
    from locate_anything.video import VideoLocator

    # 处理层方法
    expected = ["stream", "_iter_source", "_get_detect_fn"]
    for m in expected:
        assert hasattr(VideoLocator, m), f"缺少方法: {m}"

    # 不应有渲染方法
    removed = [
        "annotate", "annotate_frame",
        "_async_stream", "_render_overlay", "_draw_watermark_fast",
        "detect_and_annotate_stream", "detect_and_annotate_smooth",
        "ground_multi_and_annotate_stream", "ground_multi_and_annotate_smooth",
        "ground_single_and_annotate_smooth",
        "detect_text_and_annotate_smooth",
        "ground_gui_and_annotate_smooth",
        "point_and_annotate_smooth",
        "save_frame",
    ]
    for m in removed:
        assert not hasattr(VideoLocator, m), f"不应有渲染方法: {m}"

    print("PASS test_video_locator_structure")

# ─── 测试 6: VideoLocator.stream 模式调度 ───
def test_stream_mode_mapping():
    """测试 stream() 的 mode 参数验证。"""
    from locate_anything.video import VideoLocator

    # 构造一个虚拟 model（不加载真实模型）
    class FakeModel:
        max_edge = 512
        def detect(self, frame, categories, **kw):
            return {"answer": "", "boxes": [], "original_size": frame.size, "resized_size": frame.size, "scale_factor": 1.0}
        def ground_multi(self, frame, phrase, **kw):
            return {"answer": "", "boxes": [], "original_size": frame.size, "resized_size": frame.size, "scale_factor": 1.0}
        def detect_text(self, frame, **kw):
            return {"answer": "", "boxes": [], "original_size": frame.size, "resized_size": frame.size, "scale_factor": 1.0}
        def ground_gui(self, frame, phrase, output_type="box", **kw):
            return {"answer": "", "boxes": [], "original_size": frame.size, "resized_size": frame.size, "scale_factor": 1.0}
        def point(self, frame, phrase, **kw):
            return {"answer": "", "points": [], "original_size": frame.size, "resized_size": frame.size, "scale_factor": 1.0}

    model = FakeModel()
    vl = VideoLocator(model, frame_interval=1)

    img = Image.new("RGB", (64, 64))

    # detect 模式
    results = list(vl.stream(img, mode="detect", categories=["test"]))
    assert len(results) == 1
    assert results[0]["frame_index"] == 0
    assert "boxes" in results[0]

    # ground 模式
    results = list(vl.stream(img, mode="ground", phrase="something"))
    assert len(results) == 1

    # text 模式
    results = list(vl.stream(img, mode="text"))
    assert len(results) == 1

    # gui 模式
    results = list(vl.stream(img, mode="gui", phrase="button"))
    assert len(results) == 1

    # point 模式
    results = list(vl.stream(img, mode="point", phrase="center"))
    assert len(results) == 1
    assert "points" in results[0]

    # 无效 mode
    try:
        list(vl.stream(img, mode="invalid"))
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "未知检测模式" in str(e)

    # detect 缺少 categories
    try:
        list(vl.stream(img, mode="detect"))
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        assert "categories" in str(e)

    print("PASS test_stream_mode_mapping")

# ─── 测试 7: __init__.py 导出 ───
def test_init_exports():
    import locate_anything
    expected = [
        "VideoLocator", "VideoRenderer",
        "iter_video", "iter_camera", "iter_screen",
        "VIDEO_EXTS",
    ]
    for name in expected:
        assert hasattr(locate_anything, name), f"locate_anything 缺少导出: {name}"

    print("PASS test_init_exports")

# ─── 测试 8: VideoRenderer 类结构 ───
def test_video_renderer_structure():
    from locate_anything.rendering import (
        VideoRenderer, draw_watermark, render_overlay, annotate_frame, _get_font,
    )

    # 渲染方法
    expected = ["annotate_stream", "annotate_smooth", "_async_stream"]
    for m in expected:
        assert hasattr(VideoRenderer, m), f"VideoRenderer 缺少方法: {m}"

    # 模块级工具函数
    assert callable(draw_watermark)
    assert callable(render_overlay)
    assert callable(annotate_frame)
    assert callable(_get_font)

    # VideoRenderer 不应展开具体模式方法
    mode_methods = [
        "detect_and_annotate_stream", "detect_and_annotate_smooth",
        "ground_multi_and_annotate_stream", "ground_multi_and_annotate_smooth",
        "ground_single_and_annotate_smooth",
        "detect_text_and_annotate_smooth",
        "ground_gui_and_annotate_smooth",
        "point_and_annotate_smooth",
    ]
    for m in mode_methods:
        assert not hasattr(VideoRenderer, m), f"VideoRenderer 不应有模式特化方法: {m}"

    print("PASS test_video_renderer_structure")

# ─── 测试 9: 渲染工具函数 ───
def test_rendering_functions():
    from locate_anything.rendering import draw_watermark, render_overlay, annotate_frame, _get_font

    img = Image.new("RGB", (128, 128), color=(100, 100, 100))

    # draw_watermark 原地修改不报错
    draw_watermark(img, 0, 0.0)
    draw_watermark(img, 42, 3.5)

    # _get_font 返回字体对象（可能为默认字体）
    font = _get_font(12)
    assert font is not None
    # 缓存命中
    font2 = _get_font(12)
    assert font is font2

    # render_overlay 返回 RGBA 透明图层
    result = {
        "image": img,
        "boxes": [{"x1": 10, "y1": 10, "x2": 50, "y2": 60}],
    }
    overlay = render_overlay(result, categories=["test"])
    assert overlay.size == img.size
    assert overlay.mode == "RGBA"

    # render_overlay with points
    result_pt = {
        "image": img,
        "points": [{"x": 64, "y": 64}],
    }
    overlay_pt = render_overlay(result_pt)
    assert overlay_pt.size == img.size

    # annotate_frame 整合水印 + 标注
    class FakeModel:
        def annotate(self, img, result, **kw):
            return img

    model = FakeModel()
    annotated = annotate_frame(img, {"image": img, "frame_index": 1, "timestamp": 0.5}, model)
    assert annotated.size == img.size

    print("PASS test_rendering_functions")

# ─── 测试 10: CLI 参数解析 ───
def test_video_cli_args():
    import argparse
    # 模拟 scripts/video.py 的 parser
    parser = argparse.ArgumentParser()
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i")
    src.add_argument("--camera", type=int)
    src.add_argument("--screen", type=int)
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--mode", "-m", choices=["detect", "ground", "text", "gui", "point"], default="detect")
    parser.add_argument("--categories", "-c", default="person,face")
    parser.add_argument("--frame-interval", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--no-annotate", action="store_true")

    # 互斥组测试
    args = parser.parse_args(["-i", "test.mp4", "-c", "person"])
    assert args.input == "test.mp4"
    assert args.categories == "person"
    assert args.frame_interval == 2

    args = parser.parse_args(["--camera", "0", "-c", "face"])
    assert args.camera == 0

    args = parser.parse_args(["--screen", "1", "--frame-interval", "5"])
    assert args.screen == 1
    assert args.frame_interval == 5

    print("PASS test_video_cli_args")

# ─── 测试 11: Server 函数结构 ───
def test_server_structure():
    with open("scripts/server.py", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    funcs = {n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    required = {
        "get_model", "mjpeg_stream", "camera_stream_generator",
        "screen_stream_generator", "main",
    }
    missing = required - funcs
    assert not missing, f"Server缺少函数: {missing}"

    print("PASS test_server_structure")

# ─── 测试 12: requirements.txt 包含新依赖 ───
def test_requirements():
    with open("requirements.txt", encoding="utf-8") as f:
        content = f.read()
    assert "mss" in content, "缺少 mss"
    print("PASS test_requirements")

# ─── 运行 ───
if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    tests = [
        test_syntax,
        test_video_exts,
        test_resolve_source,
        test_video_locator_structure,
        test_stream_mode_mapping,
        test_init_exports,
        test_video_renderer_structure,
        test_rendering_functions,
        test_video_cli_args,
        test_server_structure,
        test_requirements,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            import traceback
            print(f"FAIL {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{passed}/{len(tests)} 项测试通过")
