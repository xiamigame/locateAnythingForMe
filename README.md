# 🎯 locateAnythingForMe

基于 [NVlabs/Eagle](https://github.com/NVlabs/Eagle) → **Embodied (LocateAnything-3B)** 的视觉定位 API 封装。

LocateAnything-3B 是 NVIDIA 开源的 3B 参数视觉定位模型，支持：
- 🎯 目标检测（Object Detection）
- 📝 短语定位（Phrase Grounding）
- 🔤 文字检测（OCR / Text Detection）
- 🖥️ GUI 元素定位
- 📍 指向（Pointing）

## 项目结构

```
locateAnythingForMe/
├── locate_anything/          # 主包
│   ├── __init__.py
│   ├── api.py                # 核心 API —— 封装 LocateAnythingWorker
│   └── config.py             # 配置管理
├── submodules/
│   └── Eagle/                # NVlabs/Eagle (git submodule) → 使用其中的 Embodied/
├── scripts/
│   └── img_cli.py               # 演示脚本
├── requirements.txt
├── setup.py
└── README.md
```

## 快速开始

### 1. 克隆（含子模块）

```bash
git clone --recurse-submodules <this-repo-url>
cd locateAnythingForMe
pip install -r requirements.txt
```

### 2. 使用

```python
from locate_anything import LocateAnything

la = LocateAnything(model_path="nvidia/LocateAnything-3B")

# 目标检测
result = la.detect("screenshot.jpg", ["person", "car", "dog"])
for box in result.boxes:
    print(box)  # {"x1": 100, "y1": 200, "x2": 300, "y2": 400}

# 短语定位
result = la.ground("image.jpg", "people wearing red shirts")

# 文字检测
result = la.detect_text("document.jpg")

# GUI 元素定位
result = la.ground_gui("screenshot.png", "the search button")

# 指向
result = la.point("scene.jpg", "the traffic light")
```

### 3. 命令行

```bash
python scripts/img_cli.py -i screenshot.jpg -c "person,car" -m detect
```

## 模型

模型首次使用时会自动从 HuggingFace 下载：[nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B)

- 代码许可：Apache 2.0（[Eagle 仓库](https://github.com/NVlabs/Eagle)）
- 模型权重：NVIDIA License（非商业用途）
