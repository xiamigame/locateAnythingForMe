#!/usr/bin/env python
"""
locateAnythingForMe 演示脚本

用法:
    # 目标检测 + 标注
    python scripts/demo.py -i screenshot.png -c "person,face,chair"

    # 短语定位
    python scripts/demo.py -i photo.jpg -m ground -p "people wearing red shirts"

    # 文字检测
    python scripts/demo.py -i document.jpg -m text

    # GUI 定位
    python scripts/demo.py -i screenshot.png -m gui -p "search button"

    # 指向
    python scripts/demo.py -i scene.jpg -m point -p "traffic light"

    # 调整缩放
    python scripts/demo.py -i photo.jpg -c "person" --max-edge 512
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locate_anything import LocateAnythingForMe, LocateConfig


def main():
    parser = argparse.ArgumentParser(
        description="locateAnythingForMe 演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", "-i", type=str, required=True, help="输入图像路径")
    parser.add_argument("--output", "-o", type=str, default="", help="输出图像路径（留空则自动生成）")
    parser.add_argument("--mode", "-m",
                        choices=["detect", "ground", "text", "gui", "point"],
                        default="detect", help="运行模式")
    parser.add_argument("--categories", "-c", type=str, default="person,face",
                        help="检测类别，逗号分隔（detect 模式）")
    parser.add_argument("--phrase", "-p", type=str, default="",
                        help="定位描述（ground/gui/point 模式）")
    parser.add_argument("--model", type=str, default="nvidia/LocateAnything-3B",
                        help="模型路径")
    parser.add_argument("--device", "-d", type=str, default="cuda",
                        help="推理设备 (cuda/cpu)")
    parser.add_argument("--max-edge", type=int, default=1024,
                        help="缩放最长边（512/768/1024，越小越快）")
    parser.add_argument("--no-annotate", action="store_true",
                        help="不标注，只输出文本结果")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        sys.exit(1)

    config = LocateConfig(
        model_path=args.model,
        device=args.device,
        max_edge=args.max_edge,
    )

    print(f"模型:   {config.model_path}")
    print(f"设备:   {config.device}")
    print(f"缩放:   max_edge={config.max_edge}")
    print(f"图像:   {args.image}")
    print(f"模式:   {args.mode}")
    print("-" * 50)

    la = LocateAnythingForMe(
        model_path=args.model,
        device=args.device,
        max_edge=args.max_edge,
    )

    result = None

    if args.mode == "detect":
        categories = [c.strip() for c in args.categories.split(",")]
        print(f"类别:   {categories}")
        result = la.detect(args.image, categories)

    elif args.mode == "ground":
        phrase = args.phrase or input("请输入描述: ")
        result = la.ground_multi(args.image, phrase)

    elif args.mode == "text":
        result = la.detect_text(args.image)

    elif args.mode == "gui":
        phrase = args.phrase or input("请输入 GUI 元素描述: ")
        result = la.ground_gui(args.image, phrase)

    elif args.mode == "point":
        phrase = args.phrase or input("请输入目标: ")
        result = la.point(args.image, phrase)

    if result is None:
        print("未执行任何操作")
        return

    # 输出结果
    print(f"\n原始输出: {result['answer'][:300]}...")
    boxes = result.get("boxes", [])
    points = result.get("points", [])
    if boxes:
        print(f"检测到 {len(boxes)} 个目标:")
        for i, b in enumerate(boxes):
            print(f"  [{i}] ({b['x1']:.0f}, {b['y1']:.0f}) → ({b['x2']:.0f}, {b['y2']:.0f})")
    if points:
        print(f"检测到 {len(points)} 个点:")
        for i, p in enumerate(points):
            print(f"  [{i}] ({p['x']:.0f}, {p['y']:.0f})")

    # 标注
    if not args.no_annotate:
        output_path = args.output or _default_output(args.image, args.mode)
        annotated = la.annotate(args.image, result,
                                categories=categories if args.mode == "detect" else None)
        annotated.save(output_path)
        print(f"\n标注图像已保存: {output_path}")


def _default_output(image_path: str, mode: str) -> str:
    name = os.path.basename(image_path)
    base, ext = os.path.splitext(name)
    return os.path.join("img", f"{base}_{mode}_annotated{ext}")


if __name__ == "__main__":
    main()
