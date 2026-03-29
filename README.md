# ASA (Artist Style Adherence) Benchmark

ASA Benchmark 是一套基于第一性原理设计的评估方案，专门用于量化评估 **1024 分辨率通用动漫生成模型** 对艺术家风格标签（Artist Tags）的遵从能力与泛化稳定性。

该项目由 **Kaloscope 2.0 (LSNet-XL)** 驱动，利用其 90.13% 的艺术家分类准确率作为 Ground Truth 评测环境。

---

## 🚀 快速开始

### 1. 环境准备
确保已安装必要的依赖：
```bash
pip install torch torchvision timm matplotlib numpy tqdm pillow
```

### 2. 配置说明
编辑 `benchmark_config.json`：
- `lsnet_checkpoint`: LSNet-XL 模型的权重路径。
- `class_mapping_csv`: 艺术家 ID 与名称的映射表。
- `test_artists_csv`: 您自定义的待测艺术家名单（每行一个名称）。
- `generated_images_root`: 存放生成图像的根目录。

### 3. 数据组织格式
生成的图像必须按照以下目录结构组织：
```text
generated_outputs/
├── zero_artist/          # (Baseline) 不包含艺术家标签的图像
├── standard_1girl/       # 标准提示词生成的图像
├── complex_background/   # 复杂背景提示词生成的图像
└── [other_prompt_types]/ # 其他自定义提示词类型
    └── [artist_name]/    # 每个艺术家的文件夹
        ├── sample_01.png
        └── ...
```

### 4. 运行评估
```bash
python batch_asa_eval.py --config benchmark_config.json
```

---

## 📊 数据指标解读

评估报告包含以下核心维度，帮助您深入理解模型的风格建模水平：

### 1. Weighted ASA Score (加权总分) - **核心排序指标**
- **定义**: `Overall Accuracy * Consistency Score`
- **逻辑**: 该指标同时要求模型“画得准”且“画得稳”。它会自动惩罚那些仅在特定场景下表现良好，或准确率极低但一致性高的模型。

### 2. Consistency Score (风格一致性)
- **逻辑**: 反映模型在不同提示词（简单 vs 复杂）下的风格稳定性。
- **解读**: 分数越接近 1.0，说明模型对风格的表达越解耦，不会因为加入背景或动作标签而丢失画师特征。

### 3. Style Resilience (风格韧性)
- **定义**: `Complex_Background_Acc / Standard_1girl_Acc`
- **解读**: 衡量风格在“高噪声”环境下的生存能力。如果该值远低于 1.0，说明模型对该画师的掌握仅停留在简单的肖像画层面。

### 4. Delta vs Baseline (信号增益)
- **逻辑**: 对比 `包含画师标签` 与 `不包含标签` 的准确率差异。
- **解读**: 
    - **高 Delta**: 模型受提示词强力驱动，风格信号清晰。
    - **低 Delta**: 可能是底模审美过强（数据污染），或者是模型根本没有理解该画师标签。

---

## 🧠 深度思考：如何科学评估风格？

在解读 ASA 数据时，请注意以下基于底层逻辑的洞察：

1. **Top-1 vs Top-5 的博弈**:
   如果一个模型的 Top-1 较低但 Top-5 很高，通常意味着它生成的风格处于“画师群落”的交界处（例如：京阿尼画师之间极其相似）。这并不代表模型失败，而是反映了艺术风格的连续性。

2. **拒绝“过拟合”陷阱**:
   ASA 专门针对 1024 通用模型。对于强行微调的 LoRA，虽然 ASA 得分可能极高，但往往会伴随画面质量的剧烈下降（Style Collapse）。请配合视觉审美共同评估。

3. **分辨率的归一化**:
   由于评测网络（LSNet）在 448px 下运行，它更关注**宏观特征**（线条粗细、色彩倾向、形体习惯）。如果一个模型在 1024px 下有微观笔触但在 448px 下无法被识别，说明其宏观特征抓取失败。

---

## 📂 输出结果
- `asa_benchmark_report.json`: 包含所有详细数据和汇总指标。
- `asa_benchmark_chart.png`: 自动生成的双子图看板，直观展示性能曲线与核心看板。
