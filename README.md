# ASA (Artist Style Adherence) Benchmark

ASA Benchmark 是一套基于第一性原理设计的评估方案，专门用于量化评估 **1024 分辨率通用动漫生成模型** 对艺术家风格标签（Artist Tags）的遵从能力与泛化稳定性。

该项目由 **Kaloscope 2.0 (LSNet-XL)** 驱动，利用其 90.13% 的艺术家分类准确率作为核心评测引擎。

---

## 🚀 快速开始

### 1. 环境准备
确保已安装必要的依赖：
```bash
pip install torch torchvision timm matplotlib numpy tqdm pillow requests
```

### 2. 图像生成 (ComfyUI API)
使用 `run_generation_task.py` 自动化下发生成任务。
- **环境自愈**: 脚本会自动修复 ComfyUI (Aki版) 常见的 `output\checkpoints` 路径校验错误。
- **智能队列**: 自动监控 ComfyUI 队列，当任务超过 99 个时会自动分批下发，无需人工干预。
- **标准转义**: 艺术家名称自动处理为 `artist \(name\)` 格式，确保与模型训练标签对齐。

```bash
python run_generation_task.py
```

### 3. 数据组织格式 (极简模式)
生成的图像不再需要复杂的目录结构，只需将所有测试图**直接放置**在指定的文件夹下即可：
```text
generated_outputs/
└── my_test_folder/        <-- 在 config.json 中对应的 prompt_types 名
    ├── ASA_Result_GLOBAL_BASELINE_xxxx.png  # (必选) 真正的 Baseline
    ├── ASA_Result_Mika Pikazo_xxxx.png      # 自动识别画师名
    └── ASA_Result_fuzichoco_xxxx.png        # 支持带括号、空格的智能匹配
```

### 4. 运行评估
编辑 `benchmark_config.json` 中的 `prompt_types` 为您的文件夹名，然后运行：
```bash
python batch_asa_eval.py --config benchmark_config.json
```

---

## 📊 核心指标解读 (Top-5 Focus)

在艺术风格领域，由于画师之间存在流派重叠，本项目**默认使用 Top-5 命中率**作为主评价维度。

### 1. Weighted ASA Score (加权总分)
- **定义**: `Overall Avg Top-5 * Consistency Score`
- **逻辑**: 该指标是衡量模型好坏的**第一标准**。它要求模型不仅能画出风格（高命中），还要在不同测试环境下保持稳定。

### 2. Consistency Score (风格稳定性)
- **解读**: 反映模型在不同提示词（如简单 vs 复杂）下的表现差异。分数越接近 1.0，说明模型对风格的理解越解耦，不会因为环境变化而丢失画师特征。

### 3. Delta vs Baseline (信号增益)
- **解读**: 对比“注入画师标签”与“纯净 Baseline”后的 Top-5 提升量。
    - **高 Delta**: 模型对提示词驱动响应极强。
    - **低 Delta**: 提示可能被背景标签淹没，或模型根本不认识该画师。

---

## 🧠 深度思考：底层逻辑洞察

### 1. 分辨率与特征保留 (CenterCrop vs Resize)
本项目已移除强制 448px 缩放，改为直接从 1024px 图像中进行 **CenterCrop**。
- **原理**: 缩放会使笔触平滑化，导致 LSNet 丢失关键的微观视觉特征。保留原始像素的裁剪能更真实地还原模型对线条和质感的建模能力。

### 2. 提示词“掩蔽效应”
如果您的测试得分较低，请审视您的 Baseline 提示词是否过于复杂（如包含了太多具体的衣服、饰品标签）。
- **审慎建议**: 强烈的视觉描述词会与 `artist:` 标签争夺注意力。建议在评估时对比“精简版 Prompt”与“复杂版 Prompt”的得分差异。

### 3. Top-5 的合理性
如果模型将 A 画师识别为 B 画师，但 B 恰好是 A 的模仿者或同门，Top-5 命中能有效保留这种“艺术相似性”的得分，避免了 Top-1 硬分类带来的评价偏置。

---

## 📂 输出结果
- `asa_benchmark_report.json`: 包含所有详细数据、对比指标与配置快照。
- `asa_benchmark_chart.png`: **双子图可视化看板**，左侧展示准确率分布，右侧展示核心指标统计。
