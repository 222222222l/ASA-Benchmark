# ASA Benchmark

[中文](README.md) · [English](README.en.md)

面向动漫生成图像的**画师标签一致性诊断工具**。使用 Kaloscope 2.0（LSNet）对带目标画师标签的生成图进行分类，比较不同模型或提示词条件下的命中率。它衡量生成结果与指定分类器的关联，不能直接判定真实作者、风格复现成功或整体视觉质量。

## 方法与指标

| 指标 | 当前实现与解释 |
| --- | --- |
| Top-1 / Top-5 命中率 | 目标标签是否位于分类器预测的前 1 / 5 名；只反映固定分类体系中的标签一致性 |
| 一致性分数 | `1 − std(各提示词组 Top-5 命中率)`；组间离散度的启发式摘要 |
| 加权 ASA | `提示词组平均 Top-5 × 一致性分数`；未经独立效度验证的复合指标 |
| 风格韧性 | 复杂背景与标准提示词命中率的比值；需要同时报告分子、分母及样本数 |
| 相对基线差 | 有画师条件与无画师条件的分数差；需保持模型、内容、种子和采样设置可比 |

分类器预测不是 ground truth。画师标签、构图、角色、配色和训练集重叠都可能影响分数；同源模型或数据会产生评估偏差。Top-5 命中不等于人工评审通过。

## 运行

准备匹配的 Kaloscope 2.0 权重和 `class_mapping.csv`，参考[模型发布页](https://huggingface.co/heathcliff01/Kaloscope2.0)。类别数应以具体权重与标签表一致性为准；早期文档的 39,260 与发布页后续核验的 39,261 不应混用。仓库不包含权重与完整评估数据。

1. 安装与 CUDA 匹配的 PyTorch、torchvision，以及 NumPy、Pillow、timm、matplotlib、tqdm；LSNet 的 SKA 路径还依赖 Triton。
2. 修改 [`benchmark_config.json`](benchmark_config.json) 中的本地权重、标签、图像及输出路径；现有配置是历史机器路径。
3. 按提示词类型组织图像，文件名保留脚本可解析的目标画师标签。

```bash
python batch_asa_eval.py --config benchmark_config.json
```

[`batch_asa_eval.py`](batch_asa_eval.py) 执行评分与汇总；[`run_generation_task.py`](run_generation_task.py) 是另行配置 ComfyUI 后使用的批量生成入口。当前评分预处理包含 `CenterCrop(448)`，不是完整画面评价。训练 checkpoint 可能通过 pickle 加载，只使用可信来源。

## 复现与局限

发布比较结果时，应同时保存代码提交、分类器与映射表哈希、提示词、随机种子、采样参数、逐图结果、有效/缺失样本数与类别分布。对不同画师和提示词组分别汇总，补充独立表征或盲评，并检查训练/评估图像近重复。当前仓库没有建立通用风格测量的独立效度或跨模型排行榜。

## 来源与许可

仓库代码见 [Apache-2.0](LICENSE)；分类器、上游 LSNet 实现、图像和生成模型适用各自许可。工具维护与模型方法贡献应分别署名。
