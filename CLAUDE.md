# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概述

基于 NVlabs/Eagle → Embodied (LocateAnything-3B) 的视觉定位 API 封装工具。继承官方 `LocateAnythingWorker`，增加图像预处理、坐标映射和标注功能。

## 仓库结构

```
locateAnythingForMe/
├── locate_anything/          # 主包
│   ├── __init__.py           # 导出 LocateAnythingForMe, LocateConfig
│   ├── tool.py               # ★ 核心工具（继承官方 Worker + 预处理 + 标注）
│   ├── api.py                # 薄封装 + 向后兼容别名 (LocateAnything = LocateAnythingForMe)
│   └── config.py             # 全局配置（模型、缩放策略、生成参数）
├── submodules/
│   └── Eagle/                # git submodule → 核心引用 Embodied/locateanything_worker.py
├── img/                      # 测试图片 + 标注输出（*_annotated.* 不提交）
├── docs/
│   ├── features/             # 每次迭代的功能记录（编号 + 名称）
│   └── design/               # 关键设计文档
├── scripts/
│   └── img_cli.py             # 命令行图像标注工具
├── requirements.txt
├── setup.py
└── README.md
```

## 文档规范

### 维护内容

| 文档类型 | 位置 | 说明 |
|----------|------|------|
| 迭代记录 | `docs/features/<编号>-<名称>.md` | 每次变更的方案设计，包含当前状态、变更内容、验收标准、测试计划 |
| 关键设计 | `docs/design/<设计内容>.md` | 架构设计决策（工具类设计、缩放策略、API 设计、模型选择等） |

### 更新原则

- **新增迭代**：在 `docs/features/` 下新建 `<编号>-<名称>.md`
- **设计沉淀**：迭代中产生的重要设计决策，沉淀到 `docs/design/<设计内容>.md`
- **删除内容**：功能下线时，删除对应的迭代记录和设计文档

## 环境约束

- **Python**：只能使用项目 venv（`venv/scripts/python.exe`），禁止使用系统 Python
- **工作目录**：所有产出物（文件、截图、模型缓存）必须在项目根目录内
- **Git 操作**：没有用户明确指令时，**禁止** commit 和 push
- **pip 镜像**：系统 pip 配置了清华源，但当前有 SSL 问题待解决
- **目标 Python 版本**：≥3.10（当前 venv 为 3.8，需重建）
- **模型**：`nvidia/LocateAnything-3B`（~6GB），HF 下载需设置 `HF_ENDPOINT=https://hf-mirror.com`
- **GPU**：RTX 3080 10GB，推理时图像缩放至最长边 1024 以内

## 核心架构

```
LocateAnythingForMe (tool.py)
    │
    ├── 继承 ──→ LocateAnythingWorker (官方，submodules/Eagle/Embodied/)
    │
    ├── _smart_resize()        等比缩放最长边到 max_edge，保持比例
    ├── _map_boxes_back()      模型输出 [0,1000] 归一化坐标 × 原图尺寸 = 像素坐标
    │
    ├── detect()              缩放→推理→原图坐标解析 → {answer, boxes, sizes...}
    ├── ground_multi()        多实例短语定位
    ├── ground_single()       单实例短语定位
    ├── detect_text()         文字区域检测
    ├── ground_gui()          GUI 元素定位（box/point）
    ├── point()               指向定位
    │
    ├── annotate()            在原图上绘制框/点/标签 → PIL.Image
    └── detect_and_annotate() 检测 + 标注一键完成
```

### 缩放策略

| max_edge | 显存占用 | 推理速度 | 适用场景 |
|----------|---------|---------|---------|
| 512      | ~3GB    | 最快    | 小目标密集、快速预览 |
| 768      | ~5GB    | 快      | 日常使用 |
| **1024** | ~7GB    | 平衡    | 推荐默认（RTX 3080） |

模型输出坐标是 [0, 1000] 归一化值，直接乘以原图宽高得到像素坐标，缩放不影响精度。

## 变更流程

### 步骤 1 — 方案设计（需用户确认）

1. 分析当前状态和需要变更的内容
2. 明确验收标准
3. 编写迭代记录到 `docs/features/<编号>-<名称>.md`（当前状态、变更内容、验收标准、测试计划）
4. **确认** — 用户确认后才能进入步骤 2

产出：`docs/features/<编号>-<名称>.md` + 用户签字确认。

### 步骤 2 — 构建 ↔ 评估循环（自动化，无需用户介入）

循环直到两个退出条件同时满足：

```
  构建代码 + 测试  ──→  评估（验收标准 + 完整测试套件）
         ↑                         │
         └────────── 未通过 ──────────┘
                                    ↓ 通过
                              E2E 验证（对照步骤 1 的验收标准）
                                    ↓
                              退出循环
```

- **构建者**：编写代码 + 对应测试（修改 = 更新测试，新增 = 创建新测试）
- **评估者**：对照步骤 1 的验收标准检查，运行完整测试套件，验证构建是否通过
- 未通过时：带着具体问题回到构建者继续修改
- 循环结束条件（**两者**都满足）：
  - 验收标准全部达成
  - E2E 验证通过

### 步骤 3 — 部署验证（需用户确认）

1. **重启服务** — 如有长期运行的服务则重启
2. **用户验证** — 用户在实际环境中验证变更是否正常工作
3. 如果用户发现问题 → 带着反馈回到**步骤 2**继续循环
4. **确认** — 用户确认无问题后进入步骤 4

### 步骤 4 — 收尾（无代码变更）

1. **更新文档** — 按文档规范同步更新相关文档（如有结构变化）
2. **提交推送** — 仅在用户明确指令后 commit 和 push

## 关键依赖

| 包 | 最低版本 | 说明 |
|----|---------|------|
| torch | ≥2.0.0 | 推理框架 |
| transformers | ≥4.57.1 | 模型加载（LocateAnything-3B 要求） |
| accelerate | ≥1.5.2 | 模型加速 |
| Pillow | ≥9.0.0 | 图像处理 |

子模块版本锁定在 `submodules/Eagle`，更新命令：`git submodule update --remote`
