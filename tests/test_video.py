"""
locate_anything/video.py 纯函数单元测试（无需加载模型）
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
    """所有新文件语法正确"""
    files = [
        "locate_anything/video.py",
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

# ─── 测试 4: frame_processor 装饰器 ───
def test_frame_processor_decorator():
    from locate_anything.video import frame_processor

    # 基本装饰
    @frame_processor(frame_interval=2, max_frames=3)
    def passthrough(la, frame):
        return frame

    assert passthrough.__wrapped__ is not None

    # 单张 PIL Image → 返回 1 帧
    img = Image.new("RGB", (64, 64))
    results = list(passthrough(img))
    assert len(results) == 1
    idx, ts, out_img = results[0]
    assert idx == 0
    assert out_img.size == (64, 64)

    # 无 source 且无 camera/screen → 应报错
    try:
        list(passthrough())
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

    print("PASS test_frame_processor_decorator")

# ─── 测试 5: VideoLocator 类定义 ───
def test_video_locator_structure():
    from locate_anything.video import VideoLocator

    # 检查类方法存在
    methods = [
        "detect_stream", "ground_multi_stream", "ground_single_stream",
        "detect_text_stream", "ground_gui_stream", "point_stream",
        "annotate", "annotate_frame",
        "detect_and_annotate_stream", "ground_multi_and_annotate_stream",
        "save_frame",
    ]
    for m in methods:
        assert hasattr(VideoLocator, m), f"缺少方法: {m}"

    print("PASS test_video_locator_structure")

# ─── 测试 6: __init__.py 导出 ───
def test_init_exports():
    # 检查 __init__.py 是否在 __all__ 中声明了新模块
    import locate_anything
    expected = ["VideoLocator", "frame_processor", "iter_video", "iter_camera", "iter_screen", "VIDEO_EXTS"]
    for name in expected:
        assert hasattr(locate_anything, name), f"locate_anything 缺少导出: {name}"

    print("PASS test_init_exports")

# ─── 测试 7: CLI 参数解析 ───
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

# ─── 测试 8: Server 函数结构 ───
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

# ─── 测试 9: requirements.txt 包含新依赖 ───
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
        test_frame_processor_decorator,
        test_video_locator_structure,
        test_init_exports,
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
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 项测试通过")
