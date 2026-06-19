# 002 — 视频/摄像头/屏幕实时检测

**状态**: 已完成  
**日期**: 2026-06-19  
**版本**: v0.4.0

## 变更内容

基于 `LocateAnythingForMe` 单图检测能力，新增视频流式检测系统：

### 新增文件

| 文件 | 说明 |
|------|------|
| `locate_anything/video.py` | 核心：帧源、VideoLocator、frame_processor 装饰器 |
| `scripts/video.py` | CLI：视频/摄像头/屏幕检测 |
| `scripts/server.py` | Web 前端：FastAPI + MJPEG 流（4 Tab） |
| `scripts/templates/index.html` | 前端页面 |

### 修改文件

| 文件 | 改动 |
|------|------|
| `locate_anything/__init__.py` | 导出 VideoLocator、frame_processor、帧源函数；v0.3.0→v0.4.0 |
| `locate_anything/tool.py` | HF_ENDPOINT 改为强制设置 |
| `requirements.txt` | 新增 mss |

### 删除文件

| 文件 | 原因 |
|------|------|
| `scripts/gui.py` | Gradio 前端被 FastAPI + MJPEG 替代 |

## 架构设计

### 核心原则：异步分离

```
显示线程 ──→ 全速逐帧 yield（3ms/帧，不等待检测）
检测线程 ──→ 后台处理间隔帧 → 写入共享缓存
显示线程取缓存结果 → 轻量 paste 到当前帧
```

### 帧源（生成器）

三个底层帧源，统一接口 `Iterator[(idx, ts, PIL.Image)]`：

| 帧源 | 后端 | 特点 |
|------|------|------|
| `iter_video()` | decord | 视频文件 |
| `iter_camera()` | cv2 | 实时摄像头 |
| `iter_screen()` | mss | 屏幕捕获 |

### VideoLocator 类

组合 `LocateAnythingForMe`，两种输出模式：

| 模式 | 方法 | 输出 |
|------|------|------|
| 跳帧 | `detect_and_annotate_stream()` | 仅检测帧 |
| 异步平滑 | `detect_and_annotate_smooth()` | 全部帧（显示线程+检测线程分离） |

### 性能优化

- **字体缓存**：`_get_font()` 按 size 缓存，避免每帧读磁盘
- **标注图层缓存**：检测结果预渲染为 RGBA 透明图层，非检测帧只做 paste
- **单帧源**：显示/检测共享同一个帧源实例，避免多 CameraCapture 抢设备

## 验收标准

- [x] 视频文件检测：`python scripts/video.py -i video.mp4 -c "person"`
- [x] 摄像头实时检测：`python scripts/video.py --camera 0 -c "face" --display`
- [x] Web 前端摄像头 MJPEG 流：`python scripts/server.py` → 浏览器
- [x] 前端 4 Tab 可用：摄像头/视频/屏幕/图片
- [x] 9/9 单元测试通过

## 已知限制

- Web 前端 MJPEG 帧率受浏览器限制（通常 15-30fps）
- 屏幕捕获帧率较低（~2-8fps，取决于分辨率）
- 检测速度为模型本身限制（0.5-5fps，取决于 max_edge）
