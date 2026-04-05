# Artist Style Adherence (ASA) Benchmark System

本项目是一个专为动漫生成模型设计的“艺术家风格遵从性”量化评测系统。通过集成 Kaloscope 2.0 (LSNet-XL) 视觉分类模型作为 Ground Truth 验证器，自动评估生成模型在特定画师标签下的表现。

## 核心指标体系

系统不仅关注单一的准确率，还引入了多维度的稳定性与泛化性指标：

- **Top-5 Accuracy (主指标)**: 考虑到艺术风格的重叠性（例如 A 画师是 B 画师的老师或模仿者），Top-5 命中被视为风格复现成功。
- **Weighted ASA Score (综合得分)**: `Overall Avg Top-5 * Consistency Score`。该得分惩罚了在不同 Prompt 类型下表现波动巨大的模型，反映了模型的综合素质。
- **Consistency Score (一致性评分)**: `1.0 - std(accuracies)`。衡量模型在简单背景、复杂场景等不同提示词环境下复现风格的稳定性。
- **Style Resilience (风格韧性)**: 比较“复杂背景”与“标准提示词”下的性能比率。反映了模型在干扰环境下维持艺术风格的能力。
- **Delta vs Baseline**: 评估使用特定艺术家标签后，相对于“无艺术家标签（Global Baseline）”的风格偏移量。

## 环境要求

### 硬件
- 建议使用具有 12GB+ 显存的 NVIDIA GPU (用于推理 1024x1024 图像)。

### 软件 (Linux/Windows)
- Python 3.10+
- PyTorch 2.4.1+
- [LSNet-XL 权重文件](file:///F:/ComfyUI-aki-v1.3/models/lsnet/kaloscope-2.0/best_checkpoint.pth)
- **Triton**: 在 Linux 环境下运行 SKA 算子必须安装：
  ```bash
  pip install triton
  ```

## 快速开始

1. **准备数据**:
   - 将生成的图像放置在 `generated_images_root` 目录下，按 `prompt_types` 分子目录。
   - 文件名需包含艺术家名称（例如 `ASA_Result_Mika Pikazo_00001_.png`）。

2. **配置文件**:
   - 编辑 `benchmark_config.json`，设置模型路径、数据集路径及 Prompt 类型。

3. **运行评测**:
   ```bash
   python batch_asa_eval.py --config benchmark_config.json
   ```

## 文件说明

- `batch_asa_eval.py`: 评测引擎，负责图像加载、推理、指标计算及可视化。
- `run_generation_task.py`: ComfyUI 自动化脚本，用于大规模批量生成测试图。
- `test_artists_list.csv`: 待测艺术家列表（支持 Danbooru 转义格式）。
- `class_mapping.csv`: LSNet 支持的 39,260 位艺术家映射表。

## 注意事项

- **Linux 导入报错**: 如果遇到 `No module named 'lsnet_model'`，请确保 `comfyui-lsnet` 子模块已正确下载，且脚本已将该目录添加至 `sys.path`。
- **分辨率**: 评测时会自动对图像进行 `CenterCrop(448)`，请确保生成的原始图像分辨率不低于 512px，建议 1024px。
