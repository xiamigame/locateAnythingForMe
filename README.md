# locateAnythingForMe

基于 [NVlabs/Eagle](https://github.com/NVlabs/Eagle) 的视觉定位与多模态理解 API 封装项目。

Eagle 是 NVIDIA Research 开发的一系列以视觉为中心的高分辨率多模态大语言模型（MLLM），
采用多视觉编码器混合架构（Mixture of Vision Encoders），支持高达 1K~4K 分辨率的图像理解。

本项目将其封装为易用的 Python API，方便快速集成到下游应用中。

## 项目结构

```
locateAnythingForMe/
├── locate_anything/          # 主包
│   ├── __init__.py
│   ├── api.py                # 核心 API 封装
│   └── config.py             # 配置管理
├── submodules/
│   └── Eagle/                # Eagle 子模块
├── scripts/
│   └── demo.py               # 演示脚本
├── requirements.txt
├── setup.py
└── README.md
```

## 快速开始

### 1. 克隆项目（含子模块）

```bash
git clone --recurse-submodules <this-repo-url>
cd locateAnythingForMe
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. 使用示例

```python
from locate_anything import LocateAnything

# 初始化模型
la = LocateAnything(model_path="NVEagle/Eagle-X5-13B-Chat")

# 分析图像
result = la.describe("path/to/image.jpg", prompt="Describe this image.")
print(result)

# 定位目标
results = la.locate("path/to/image.jpg", target="a red car")
for item in results:
    print(item)
```

## License

本项目代码采用 Apache 2.0 许可。Eagle 模型权重使用 CC BY-NC 4.0 许可（仅限非商业用途）。
