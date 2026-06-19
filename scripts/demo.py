#!/usr/bin/env python
"""
locateAnythingForMe 演示脚本 —— 截图测试
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locate_anything import LocateAnything, LocateConfig


def main():
    parser = argparse.ArgumentParser(description="locateAnythingForMe 演示")
    parser.add_argument("--image", "-i", type=str, required=True, help="输入图像路径")
    parser.add_argument("--categories", "-c", type=str, default="person,car,bicycle",
                        help="检测类别，逗号分隔")
    parser.add_argument("--model", "-m", type=str, default="nvidia/LocateAnything-3B", help="模型路径")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="推理设备")
    parser.add_argument("--mode", choices=["detect", "ground", "text", "gui", "point"],
                        default="detect", help="运行模式")
    parser.add_argument("--phrase", "-p", type=str, default="", help="定位/指向的描述")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        sys.exit(1)

    config = LocateConfig(
        model_path=args.model,
        device=args.device,
    )

    print(f"模型: {config.model_path}")
    print(f"设备: {config.device}")
    print(f"图像: {args.image}")
    print(f"模式: {args.mode}")
    print("-" * 50)

    la = LocateAnything(config)

    if args.mode == "detect":
        categories = [c.strip() for c in args.categories.split(",")]
        print(f"检测类别: {categories}")
        result = la.detect(args.image, categories)
        print(f"\n原始输出:\n{result.raw_output}")
        print(f"\n解析结果 ({len(result.boxes)} 个框):")
        for i, box in enumerate(result.boxes):
            print(f"  [{i}] x1={box['x1']:.0f} y1={box['y1']:.0f} "
                  f"x2={box['x2']:.0f} y2={box['y2']:.0f}")

    elif args.mode == "ground":
        phrase = args.phrase or input("请输入描述: ")
        result = la.ground(args.image, phrase)
        print(f"\n结果:\n{result.raw_output}")

    elif args.mode == "text":
        result = la.detect_text(args.image)
        print(f"\n结果:\n{result.raw_output}")

    elif args.mode == "gui":
        phrase = args.phrase or input("请输入元素描述: ")
        result = la.ground_gui(args.image, phrase)
        print(f"\n结果:\n{result.raw_output}")

    elif args.mode == "point":
        phrase = args.phrase or input("请输入目标描述: ")
        result = la.point(args.image, phrase)
        print(f"\n结果:\n{result.raw_output}")


if __name__ == "__main__":
    main()
