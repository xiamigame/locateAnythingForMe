# 架构设计

## 继承链

```
LocateAnythingWorker (官方, submodules/Eagle/Embodied/locateanything_worker.py)
    │
    └── LocateAnythingForMe (locate_anything/tool.py)
            │
            ├── 新增：_smart_resize()        图像等比缩放
            ├── 新增：annotate()            结果可视化标注
            ├── 新增：detect_and_annotate() 检测+标注一站式
            │
            ├── 重写：detect()              resize → super.detect() → 原图坐标解析
            ├── 重写：ground_multi()        resize → super.ground_multi() → 原图坐标解析
            ├── 重写：ground_single()       resize → super.ground_single() → 原图坐标解析
            ├── 重写：detect_text()         resize → super.detect_text() → 原图坐标解析
            ├── 重写：ground_gui()          resize → super.ground_gui() → 原图坐标解析
            └── 重写：point()               resize → super.point() → 原图坐标解析
```

## 设计决策

### 为什么用继承而非组合

- 官方 `LocateAnythingWorker` 是对 `transformers.AutoModel` 的完整封装，包含模型加载、推理、batch runtime 等
- 继承可以零成本使用官方的所有方法和后续更新
- 只需在关键推理方法上做预处理/后处理即可

### 为什么坐标不需要映射

模型输出的 bounding box 坐标是 [0, 1000] 的**归一化值**，与输入图像的像素尺寸无关。因此缩放图像后再推理，只需在解析时用**原图尺寸**计算像素坐标即可，无需对模型输出做任何数学变换。

```
box_pixel_x = box_normalized_x / 1000 * original_image_width
```

### 为什么要缓存 scale_factor 等信息

每个方法返回 dict 包含 `original_size`、`resized_size`、`scale_factor`，便于：
- 调试时查看缩放比例
- 后续如果需要在缩放图上绘制再放大时使用
- 日志记录与性能分析
