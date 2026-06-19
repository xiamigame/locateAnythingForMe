# API 设计

## 初始化

```python
la = LocateAnythingForMe(
    model_path="nvidia/LocateAnything-3B",  # HF ID 或本地路径
    device="cuda",                          # cuda / cpu
    dtype=torch.bfloat16,                   # 模型精度
    max_edge=1024,                          # 缩放最长边
    min_edge=None,                          # 最短边下限
)
```

## 推理方法

所有推理方法遵循统一模式：`resize → super().method() → parse on original`

| 方法 | 用途 | 输入 | 返回 |
|------|------|------|------|
| `detect(image, categories)` | 目标检测 | 图 + 类别列表 | `{answer, boxes, original_size, resized_size, scale_factor}` |
| `ground_multi(image, phrase)` | 多实例定位 | 图 + 描述 | 同上 |
| `ground_single(image, phrase)` | 单实例定位 | 图 + 描述 | 同上 |
| `detect_text(image)` | 文字检测 | 图 | 同上 |
| `ground_gui(image, phrase, output_type)` | GUI 定位 | 图 + 描述 + "box"/"point" | 同上 + points |
| `point(image, phrase)` | 指向 | 图 + 描述 | `{answer, points, ...}` |

### 返回结构

```python
{
    "answer": str,              # 模型原始输出文本
    "boxes": [                  # 解析后的框（原图像素坐标）
        {"x1": float, "y1": float, "x2": float, "y2": float},
        ...
    ],
    "points": [                 # 解析后的点（point 模式）
        {"x": float, "y": float},
        ...
    ],
    "original_size": (w, h),    # 原图尺寸
    "resized_size": (w, h),     # 缩放后尺寸
    "scale_factor": float,      # 缩放倍率（原图/缩放图）
}
```

## 标注方法

```python
annotated = la.annotate(image, result, categories=["person", "car"])
# → PIL.Image（原图分辨率，绘制了框和标签）
```

参数：
- `line_width` — 自动根据图像大小计算
- `font_size` — 自动根据图像大小计算
- `colors` — 循环使用的颜色列表
- `categories` — 标签文本列表

## 一站式方法

```python
result, annotated = la.detect_and_annotate("photo.jpg", ["person", "car"])
```

等价于 `detect()` + `annotate()`，一步完成。

## 向后兼容

```python
from locate_anything import LocateAnything  # = LocateAnythingForMe 的别名
```

`api.py` 中 `LocateAnything = LocateAnythingForMe`，旧代码无需修改。
