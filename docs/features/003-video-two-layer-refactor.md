# 003 — 视频模块两层拆分 + 方法合并

**状态**: 已完成  
**日期**: 2026-06-19  
**版本**: v0.4.1

## 变更动机

002 的 `video.py` 782 行混合了处理（帧源、检测调度）和渲染（标注、水印、叠加层），导致：
- `VideoLocator` 14 个方法，6 个模式 × 3 种输出方式，大量 boilerplate
- 渲染逻辑无法独立复用
- `frame_processor` 装饰器与 `VideoLocator` 功能重叠，无实际调用方

## 变更内容

### 新建文件

| 文件 | 说明 |
|------|------|
| `locate_anything/rendering.py` | 渲染层：`VideoRenderer`（2 方法） + `draw_watermark`/`render_overlay`/`annotate_frame` |

### 修改文件

| 文件 | 改动 |
|------|------|
| `locate_anything/video.py` | 重写：纯处理层，`VideoLocator.stream()` 合并 6 个模式方法，移除所有渲染代码 |
| `locate_anything/__init__.py` | 新增 `VideoRenderer` 导出，移除 `frame_processor` |
| `scripts/video.py` | 适配新 API + 新增 `--smooth` 异步平滑模式 |
| `scripts/server.py` | 适配 `VideoRenderer` + `GeneratorExit` 客户端断开清理 |
| `tests/test_video.py` | 新增渲染层测试 + 模式调度测试，移除 frame_processor 测试 |

### 删除内容

| 删除项 | 原因 |
|--------|------|
| `frame_processor` 装饰器 | 功能被 `VideoLocator + VideoRenderer` 覆盖，无实际调用方 |
| `VideoLocator.save_frame()` | 仅 2 行代码，调用方内联即可 |
| `VideoLocator` 14 个方法 | 合并为 1 个 `stream()` + `VideoRenderer` 2 个 |

## 架构变更

### 之前（002）

```
video.py (782 行)
├── 帧源生成器 ×3
├── VideoLocator
│   ├── 6 个 *_stream         (处理)
│   ├── 8 个 *_and_annotate_* (混合)
│   └── 标注辅助方法          (渲染)
└── frame_processor 装饰器    (重复功能)
```

### 之后（003）

```
video.py (~250 行)                    rendering.py (~250 行)
├── 帧源生成器 ×3                     ├── 标注工具函数 ×4
├── VideoLocator                      └── VideoRenderer
│   └── stream(mode=...)  ×1              ├── annotate_stream()
└── （无渲染代码）                        └── annotate_smooth()
```

**方法数：14 → 3**

### 调用方式对比

```python
# 之前: 模式 × 输出方式 = 14 个方法
vl.detect_stream(source, categories)
vl.detect_and_annotate_stream(source, categories)
vl.detect_and_annotate_smooth(source, categories)
# ... × 6 种模式

# 之后: 处理 + 渲染组合
vr = VideoRenderer(vl)                              # 1 次创建
vl.stream(source, mode="detect", categories=cats)    # 1 个检测方法
vr.annotate_stream(stream, categories=cats)          # 同步标注
vr.annotate_smooth(source, detect_fn=..., ...)       # 异步平滑
```

## Bug 修复

| 问题 | 修复 |
|------|------|
| 异步帧回滚（叠加层绑定旧帧） | 缓存检测结果 dict，每帧基于当前画面实时渲染 |
| 客户端断开后 GPU 仍满载 | 清空队列 + 非阻塞 poison pill，worker 最多 1 次额外推理 |
| 模型 Statistic Info 刷屏 | Worker 线程内 redirect_stdout 到 os.devnull |
| Ctrl+C 后进程无法退出 | 移除 `worker.join()` 阻塞调用 |

## 测试

```bash
venv/python.exe tests/test_video.py    # 11/11
```

## 验收标准

- [x] `VideoLocator` 无 PIL 渲染代码，仅 `stream()` 一个方法
- [x] `VideoRenderer` 仅 `annotate_stream()` + `annotate_smooth()` 两个方法
- [x] CLI 支持 `--smooth` 异步平滑模式
- [x] Web server 支持客户端断开后资源清理
- [x] 11/11 单元测试通过
