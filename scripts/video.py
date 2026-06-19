#!/usr/bin/env python
"""
locateAnythingForMe 视频标注命令行工具

用法:
    # 视频文件目标检测
    python scripts/video.py -i video.mp4 -c "person,car" --frame-interval 2 -o output/

    # 摄像头实时检测
    python scripts/video.py --camera 0 -c "face,person" --frame-interval 2

    # 屏幕实时检测
    python scripts/video.py --screen 1 -c "button,icon" --frame-interval 5

    # 短语定位
    python scripts/video.py -i video.mp4 -m ground -p "people walking"

    # 文字检测
    python scripts/video.py -i video.mp4 -m text

    # 实时显示（cv2 窗口）
    python scripts/video.py --camera 0 -c "person" --display
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import locate_anything  # noqa: F401  环境初始化在 __init__.py 里

import argparse
import logging
import os
import signal

from locate_anything import LocateAnythingForMe, VideoLocator, VideoRenderer, LocateConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")


def main():
    parser = argparse.ArgumentParser(
        description="locateAnythingForMe 视频检测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # 输入源（三选一）
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--input", "-i", type=str, help="视频文件路径")
    src_group.add_argument("--camera", type=int, help="摄像头编号（0=默认）")
    src_group.add_argument("--screen", type=int, help="屏幕 monitor 编号（1=主显示器）")

    # 输出
    parser.add_argument("--output", "-o", type=str, default="output",
                        help="输出目录（默认 output/）")
    parser.add_argument("--display", action="store_true",
                        help="实时显示检测结果（cv2 窗口，Ctrl+C / Q 退出）")

    # 检测模式
    parser.add_argument("--mode", "-m",
                        choices=["detect", "ground", "text", "gui", "point"],
                        default="detect", help="检测模式")

    # 检测参数
    parser.add_argument("--categories", "-c", type=str, default="person,face",
                        help="检测类别，逗号分隔（detect 模式）")
    parser.add_argument("--phrase", "-p", type=str, default="",
                        help="定位描述（ground/gui/point 模式）")
    parser.add_argument("--output-type", type=str, default="box",
                        choices=["box", "point"],
                        help="GUI 定位输出类型（gui 模式）")

    # 帧控制
    parser.add_argument("--frame-interval", type=int, default=2,
                        help="帧间隔：每 N 帧取 1 帧（默认 2）")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="最大处理帧数")

    # 模型参数
    parser.add_argument("--model", type=str, default="nvidia/LocateAnything-3B")
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument("--max-edge", type=int, default=1024,
                        help="缩放最长边（512/768/1024）")

    # 标注
    parser.add_argument("--no-annotate", action="store_true",
                        help="不标注，只输出文本结果")
    parser.add_argument("--smooth", action="store_true",
                        help="异步平滑模式：后台间隔检测，全帧率输出（需 --display 实时查看）")

    # 生成参数
    parser.add_argument("--generation-mode", default="fast",
                        choices=["fast", "slow", "hybrid"],
                        help="生成模式（fast=MTP多token, slow=NTP逐token, hybrid=混合）")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度（0=贪心解码最快）")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="最大生成 token 数")

    args = parser.parse_args()

    if args.smooth and args.no_annotate:
        parser.error("--smooth 需要标注渲染，不能与 --no-annotate 同时使用")

    # ── 初始化模型 ──────────────────────────────────
    print(f"模型:   {args.model}")
    print(f"设备:   {args.device}")
    print(f"缩放:   max_edge={args.max_edge}")
    print(f"帧间隔: {args.frame_interval}")
    print(f"模式:   {args.mode}")
    print("-" * 50)

    config = LocateConfig(
        model_path=args.model,
        device=args.device,
        max_edge=args.max_edge,
        generation_mode=args.generation_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )
    la = LocateAnythingForMe(config=config)
    vl = VideoLocator(la, frame_interval=args.frame_interval)

    # ── 确定输入源 ──────────────────────────────────
    if args.camera is not None:
        source = args.camera  # int → iter_camera
        source_label = f"摄像头 #{args.camera}"
    elif args.screen is not None:
        source = f"screen:{args.screen}"
        source_label = f"屏幕 #{args.screen}"
    else:
        source = args.input
        source_label = f"视频 {args.input}"

    print(f"输入源: {source_label}")

    # ── 类别 ────────────────────────────────────────
    categories = [c.strip() for c in args.categories.split(",")] if args.categories else []
    if categories and args.mode == "detect":
        print(f"类别:   {categories}")

    # ── 构建检测参数 ────────────────────────────────
    stream_kwargs = {"mode": args.mode, "max_frames": args.max_frames}
    annotate_kw = {}  # 标注参数

    if args.mode == "detect":
        stream_kwargs["categories"] = categories
        annotate_kw["categories"] = categories
    elif args.mode == "ground":
        phrase = args.phrase or input("请输入描述: ")
        stream_kwargs["phrase"] = phrase
    elif args.mode == "text":
        pass
    elif args.mode == "gui":
        phrase = args.phrase or input("请输入 GUI 元素描述: ")
        stream_kwargs["phrase"] = phrase
        stream_kwargs["output_type"] = args.output_type
    elif args.mode == "point":
        phrase = args.phrase or input("请输入目标: ")
        stream_kwargs["phrase"] = phrase

    vr = VideoRenderer(vl) if not args.no_annotate else None

    # smooth 模式需要检测函数（直接用 model，绕过 VideoLocator.stream）
    if args.smooth:
        if args.mode == "detect":
            detect_fn = lambda f: la.detect(f, categories)
        elif args.mode == "ground":
            detect_fn = lambda f: la.ground_multi(f, phrase)
        elif args.mode == "text":
            detect_fn = lambda f: la.detect_text(f)
        elif args.mode == "gui":
            detect_fn = lambda f: la.ground_gui(f, phrase, output_type=args.output_type)
        elif args.mode == "point":
            detect_fn = lambda f: la.point(f, phrase)

    # ── 实时显示窗口 ────────────────────────────────
    display_window = None
    if args.display:
        import cv2
        display_window = f"LocateAnything — {source_label}"
        cv2.namedWindow(display_window, cv2.WINDOW_NORMAL)
        print(f"实时显示窗口: {display_window}（按 Q 退出）")

    # ── 主循环 ──────────────────────────────────────
    frame_count = 0
    interrupted = False

    def on_interrupt(sig, frame):
        nonlocal interrupted
        interrupted = True
        print("\n收到中断信号，正在退出...")

    signal.signal(signal.SIGINT, on_interrupt)

    try:
        if args.no_annotate:
            # 纯文本输出
            for result in vl.stream(source, **stream_kwargs):
                idx = result["frame_index"]
                ts = result["timestamp"]
                boxes = result.get("boxes", [])
                points = result.get("points", [])
                print(f"\n--- 帧 #{idx} ({ts:.1f}s) ---")
                if boxes:
                    print(f"  检测到 {len(boxes)} 个目标:")
                    for j, b in enumerate(boxes):
                        print(f"    [{j}] ({b['x1']:.0f}, {b['y1']:.0f}) → ({b['x2']:.0f}, {b['y2']:.0f})")
                if points:
                    print(f"  检测到 {len(points)} 个点:")
                    for j, p in enumerate(points):
                        print(f"    [{j}] ({p['x']:.0f}, {p['y']:.0f})")
                frame_count += 1

        elif args.smooth:
            # 异步平滑：后台间隔检测 + 全帧率渲染
            print(f"平滑模式: 每 {args.frame_interval} 帧检测一次，全帧率输出")
            for idx, ts, annotated, result in vr.annotate_smooth(
                source,
                detect_fn=detect_fn,
                detect_interval=args.frame_interval,
                max_frames=args.max_frames,
                **annotate_kw,
            ):
                frame_count += 1
                boxes = result.get("boxes", []) if result else []
                status = f"{len(boxes)} obj" if result else "(缓存)"
                print(f"\r帧 #{idx:06d} ({ts:.1f}s) | {status}", end="", flush=True)

                if display_window:
                    import cv2
                    import numpy as np
                    frame_bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
                    cv2.imshow(display_window, frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("\n用户关闭窗口")
                        break

                if interrupted:
                    break
            print()  # 换行

        else:
            # 同步标注：检测 + 标注 + 保存
            stream = vl.stream(source, **stream_kwargs)
            for idx, ts, annotated, result in vr.annotate_stream(stream, **annotate_kw):
                os.makedirs(args.output, exist_ok=True)
                path = os.path.join(args.output, f"frame_{idx:06d}.jpg")
                annotated.save(path)
                boxes = result.get("boxes", [])
                print(f"帧 #{idx:06d} ({ts:.1f}s) | {len(boxes)} 目标 | {os.path.basename(path)}")
                frame_count += 1

                if display_window:
                    import cv2
                    import numpy as np
                    frame_bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
                    cv2.imshow(display_window, frame_bgr)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:
                        print("用户关闭窗口")
                        break

                if interrupted:
                    break

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        if display_window:
            import cv2
            cv2.destroyAllWindows()

    print(f"\n完成。共处理 {frame_count} 帧，输出到 {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
