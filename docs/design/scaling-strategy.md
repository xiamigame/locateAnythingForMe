# 缩放策略

## 问题

RTX 3080 10GB 显存，LocateAnything-3B 模型约 6GB。原始截图通常 2560×1440 甚至 4K，直接送入模型会：
- 显存溢出（高分辨率图像 token 数量暴增）
- 推理速度极慢

## 方案

### 等比缩放

```python
def _smart_resize(image, max_edge=1024, min_edge=None):
    orig_w, orig_h = image.size
    longest = max(orig_w, orig_h)

    if longest <= max_edge:
        return image, 1.0, (orig_w, orig_h)  # 无需缩放

    scale = max_edge / longest
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.LANCZOS)
    return resized, 1.0 / scale, (orig_w, orig_h)
```

### 缩放档位

| max_edge | 典型尺寸 | 显存 | 适用场景 |
|----------|---------|------|---------|
| 512      | 512×288 | ~3GB | 小目标密集、快速预览、低显存设备 |
| 768      | 768×432 | ~5GB | 日常使用，精度与速度平衡 |
| **1024** | 1024×576 | ~7GB | 推荐默认，RTX 3080 稳定运行 |

### LANCZOS 重采样

选择 `Image.LANCZOS` 因为：
- 高质量下采样，保留更多细节
- 比 BICUBIC 稍慢但结果更清晰，对检测精度有利

## 坐标精度

模型输出坐标是 [0, 1000] 归一化值，与原图分辨率无关。

```
缩放前：原图 2560×1440 → 检测 box [500, 300, 800, 600]（归一化）
→ pixel_x1 = 500/1000 × 2560 = 1280px

缩放后：缩放图 1024×576 → 检测 box [500, 300, 800, 600]（归一化，几乎相同）
→ pixel_x1 = 500/1000 × 2560 = 1280px（用原图尺寸计算，结果一致）
```

**结论：缩放不影响坐标精度**，因为模型学习的是归一化的空间关系。
