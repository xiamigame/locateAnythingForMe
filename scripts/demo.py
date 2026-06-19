#!/usr/bin/env python
"""
locateAnythingForMe 演示脚本
"""
import argparse
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locate_anything import LocateAnything, LocateConfig


def main():
    parser = argparse.ArgumentParser(description="locateAnythingForMe 演示")
    parser.add_argument("--image", "-i", type=str, required=True, help="输入图像路径")
    parser.add_argument("--prompt", "-p", type=str, default="Describe this image in detail.", help="文本提示")
    parser.add_argument("--model", "-m", type=str, default="NVEagle/Eagle-X5-13B-Chat", help="模型路径")
    parser.add_argument("--device", "-d", type=str, default="cuda", help="推理设备 (cuda/cpu)")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.2, help="采样温度")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"错误: 图像文件不存在: {args.image}")
        sys.exit(1)

    config = LocateConfig(
        model_path=args.model,
        device=args.device,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    print(f"模型: {config.model_path}")
    print(f"设备: {config.device}")
    print(f"图像: {args.image}")
    print(f"提示: {args.prompt}")
    print("-" * 50)

    la = LocateAnything(config)
    result = la.describe(args.image, prompt=args.prompt)

    print(f"\n结果:\n{result.output}")


if __name__ == "__main__":
    main()
