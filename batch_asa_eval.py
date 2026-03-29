import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

print("Modules imported. Initializing environment...")

# 确保能加载项目中的 lsnet 模型组件
# 将 comfyui-lsnet 加入 Python 路径
current_dir = Path(__file__).parent.absolute()
lsnet_path = str(current_dir / "comfyui-lsnet")
if lsnet_path not in sys.path:
    sys.path.append(lsnet_path)

print(f"Current dir: {current_dir}")
print(f"LSNet path: {lsnet_path}")

try:
    print("Loading torch...")
    import torch
    print("Loading torch.nn.functional...")
    import torch.nn.functional as F
    print("Loading PIL.Image...")
    from PIL import Image
    print("Loading torchvision.transforms...")
    from torchvision import transforms
    print("Loading timm.models...")
    from timm.models import create_model
    print("Loading lsnet_model...")
    from lsnet_model import lsnet_artist  # noqa: F401
    print("Core deep learning libraries loaded.")
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"UNEXPECTED ERROR DURING IMPORT: {e}")
    sys.exit(1)

def load_config(config_path: str):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def sanitize_name(name: str, replace_underscore: bool) -> str:
    """处理艺术家名称转义：下划线转空格，括号转义 (幂等处理)"""
    if replace_underscore:
        name = name.replace('_', ' ')
    # 统一转义括号，先反转义再重新转义以保证幂等
    name = name.replace(r'\(', '(').replace(r'\)', ')')
    name = name.replace('(', r'\(').replace(')', r'\)')
    return name

def unescape_name(name: str) -> str:
    """反转义名称，用于文件路径匹配"""
    return name.replace(r'\(', '(').replace(r'\)', ')')

def load_mappings(csv_path: str, replace_underscore: bool) -> Tuple[Dict[str, int], Dict[int, str]]:
    """加载 class_mapping.csv，并建立映射。为了兼容性，键名使用下划线格式。"""
    name_to_id = {}
    id_to_name = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row['class_name']
            class_id = int(row['class_id'])
            # 无论输入是什么格式，映射表内部统一使用原始下划线格式
            name_to_id[raw_name] = class_id
            id_to_name[class_id] = raw_name
    return name_to_id, id_to_name

def load_test_artists(csv_path: str, replace_underscore: bool) -> List[str]:
    """加载待测试的艺术家列表"""
    artists = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row:
                artists.append(row[0].strip())
    return artists

def get_mapping_key(artist_name: str) -> str:
    """将测试列表中的转义名称还原为 mapping 表中的原始下划线格式"""
    # 1. 还原括号转义
    name = artist_name.replace(r'\(', '(').replace(r'\)', ')')
    # 2. 空格转回下划线
    name = name.replace(' ', '_')
    return name

def get_asa_model(checkpoint_path: str, num_classes: int, device: str):
    """加载评测模型"""
    # 强制指定特征维度为 2048，匹配 LSNet-XL 的标准输出
    model = create_model('lsnet_xl_artist_448', pretrained=False, num_classes=num_classes, feature_dim=2048)
    
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    # 归一化权重键名
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 容错加载：支持 Kaloscope 2.0 可能存在的 projection 层差异
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

def generate_visualization(summary: Dict, metrics: Dict, report_dir: Path, baseline_pt: str = None):
    """生成 Benchmark 结果的可视化图表 (双子图：柱状图 + 指标摘要)"""
    prompt_types = list(summary.keys())
    top1_accs = [m['top1_accuracy'] * 100 for m in summary.values()]
    top5_accs = [m['top5_accuracy'] * 100 for m in summary.values()]
    
    x = np.arange(len(prompt_types))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [2, 1]})
    
    # 子图 1: 准确率柱状图
    rects1 = ax1.bar(x - width/2, top1_accs, width, label='Top-1 Accuracy (%)', color='#3498db')
    rects2 = ax1.bar(x + width/2, top5_accs, width, label='Top-5 Accuracy (%)', color='#2ecc71')

    if baseline_pt and baseline_pt in summary:
        baseline_val = summary[baseline_pt]['top1_accuracy'] * 100
        ax1.axhline(y=baseline_val, color='r', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_pt})')

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('ASA Performance by Prompt Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(prompt_types, rotation=15)
    ax1.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax1.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)

    # 子图 2: 核心指标汇总 (雷达图/文本看板)
    ax2.axis('off')
    metrics_text = (
        f"Core Metrics Summary (Top-5 Focus)\n"
        f"{'-'*35}\n"
        f"Overall Avg Top-5: {metrics['overall_avg_acc']:.2%}\n"
        f"Consistency Score: {metrics['consistency_score']:.4f}\n"
        f"Delta vs Baseline: {metrics['delta_vs_baseline']:+.4f}\n"
        f"Style Resilience:  {metrics['style_resilience']:.4f}\n"
        f"{'-'*35}\n"
        f"Weighted Score:   {metrics['weighted_score']:.4f}\n"
        f"(Score = Avg_Top5 * Consistency)"
    )
    ax2.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace', va='center')

    fig.tight_layout()
    plt.savefig(report_dir / "asa_benchmark_chart.png")
    plt.close()

def main():
    print("Initializing ASA Benchmark Evaluation...")
    parser = argparse.ArgumentParser(description="ASA Benchmark Batch Evaluation")
    parser.add_argument("--config", default="benchmark_config.json", help="Path to config file")
    args = parser.parse_args()

    # 1. 加载配置与数据
    config = load_config(args.config)
    eval_cfg = config['evaluation_settings']
    data_cfg = config['test_data_settings']
    out_cfg = config['output_settings']

    replace_underscore = data_cfg['name_sanitization']['replace_underscore_with_space']
    baseline_pt = data_cfg.get('baseline_prompt_type', None)
    
    name_to_id, id_to_name = load_mappings(eval_cfg['class_mapping_csv'], replace_underscore)
    test_artists = load_test_artists(data_cfg['test_artists_csv'], replace_underscore)
    
    print(f"Loaded {len(name_to_id)} artists from mapping.")
    print(f"Targeting {len(test_artists)} artists for benchmark.")

    model = get_asa_model(eval_cfg['lsnet_checkpoint'], len(id_to_name), eval_cfg['device'])

    # 2. 图像预处理 (移除强制 448 缩放，改为仅 CenterCrop 保持特征)
    # LSNet 内部会自行处理输入尺寸，外部预处理过大会导致特征平滑
    transform = transforms.Compose([
        # 移除 transforms.Resize(448),
        transforms.CenterCrop(448),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 3. 遍历 Prompt Types 和 Artists 进行评测
    image_root = Path(data_cfg['generated_images_root'])
    prompt_types = data_cfg['prompt_types']
    
    stats_per_prompt = {pt: {"correct": 0, "total": 0, "top5": 0, "conf_sum": 0} for pt in prompt_types}
    detailed_results = []

    for pt in prompt_types:
        print(f"\nEvaluating Prompt Type: {pt}")
        pt_dir = image_root / pt
        if not pt_dir.exists():
            print(f"[Warning] Prompt directory {pt_dir} not found. Skipping.")
            continue

        for artist_name in tqdm(test_artists, desc=f"Artists in {pt}"):
            # 1. 获取映射表中的键名 (原始下划线格式)
            mapping_key = get_mapping_key(artist_name)
            
            # 2. 清洗出 safe_name (对应 run_generation_task.py 中的逻辑，用于文件名匹配)
            safe_name = re.sub(r'[^\w\s\(\)\-]', '', artist_name).strip()
            
            # 3. 在目录中寻找匹配该艺术家 safe_name 的所有文件
            pattern = f"ASA_Result_{pt}_*{safe_name}*_*.png"
            image_paths = list(pt_dir.glob(pattern))
            
            if not image_paths:
                image_paths = list(pt_dir.glob(f"*{safe_name}*.png"))
            
            if not image_paths:
                base_name = unescape_name(artist_name)
                image_paths = list(pt_dir.glob(f"*{base_name}*.png"))

            if image_paths:
                if mapping_key not in name_to_id:
                    print(f"Warning: Found files for {artist_name} (key: {mapping_key}) but artist not in mapping.")
                    continue
            else:
                continue

            expected_id = name_to_id[mapping_key]

            # 批量推理
            for i in range(0, len(image_paths), eval_cfg['batch_size']):
                batch_paths = image_paths[i:i+eval_cfg['batch_size']]
                try:
                    imgs = torch.stack([transform(Image.open(p).convert('RGB')) for p in batch_paths]).to(eval_cfg['device'])
                except Exception as e:
                    print(f"Error loading images in {artist_dir}: {e}")
                    continue

                with torch.no_grad():
                    logits = model(imgs)
                    probs = F.softmax(logits, dim=-1)
                    top5_probs, top5_indices = torch.topk(probs, k=5, dim=-1)
                
                for idx_in_batch, p in enumerate(batch_paths):
                    pred_ids = [int(x) for x in top5_indices[idx_in_batch].tolist()]
                    target_id = int(expected_id)
                    
                    is_top1 = (pred_ids[0] == target_id)
                    is_top5 = (target_id in pred_ids)
                    conf = float(top5_probs[idx_in_batch][0])

                    # --- DEBUG 打印：每类画师仅打印第一个样本的对比 ---
                    if i == 0 and idx_in_batch == 0:
                        print(f"\n[DEBUG] File: {p.name}")
                        print(f"        Expected: {id_to_name[expected_id]} (ID: {expected_id})")
                        print(f"        Top-3 Predictions:")
                        for rank in range(3):
                            p_id = pred_ids[rank]
                            p_name = id_to_name.get(p_id, "Unknown")
                            p_conf = float(top5_probs[idx_in_batch][rank])
                            print(f"          {rank+1}. {p_name} (ID: {p_id}, Conf: {p_conf:.4f})")

                    stats_per_prompt[pt]["total"] += 1
                    if is_top1: stats_per_prompt[pt]["correct"] += 1
                    if is_top5: stats_per_prompt[pt]["top5"] += 1
                    stats_per_prompt[pt]["conf_sum"] += conf

                    if out_cfg['save_detailed_json']:
                        detailed_results.append({
                            "prompt_type": pt,
                            "artist": artist_name,
                            "file": str(p.relative_to(image_root)),
                            "top1_pred": id_to_name[top5_indices[idx_in_batch][0].item()],
                            "confidence": conf,
                            "is_correct": bool(is_top1),
                            "is_top5": bool(is_top5)
                        })

    # 4. 生成报告与指标计算
    report_dir = Path(out_cfg['report_dir'])
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    non_baseline_accs = []
    
    for pt, stats in stats_per_prompt.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"]
            summary[pt] = {
                "top1_accuracy": acc,
                "top5_accuracy": stats["top5"] / stats["total"],
                "avg_confidence": stats["conf_sum"] / stats["total"],
                "sample_count": stats["total"]
            }
            if pt != baseline_pt:
                non_baseline_accs.append(acc)

    # --- 核心指标计算优化 (已切换至 Top-5 作为主指标) ---
    # 1. Consistency Score (1 - std) - 使用 Top-5 计算
    non_baseline_top5_accs = [m['top5_accuracy'] for pt, m in summary.items() if pt != baseline_pt]
    
    # 逻辑容错：如果仅有 Baseline 数据，则不进行排除
    if not non_baseline_top5_accs and baseline_pt in summary:
        non_baseline_top5_accs = [summary[baseline_pt]['top5_accuracy']]
        print("[Notice] Only baseline data found. Metrics will be calculated based on baseline.")

    consistency_score = 1.0 - np.std(non_baseline_top5_accs) if len(non_baseline_top5_accs) > 1 else 1.0
    
    # 2. Overall Avg Top-5 Acc
    overall_avg_top5_acc = np.mean(non_baseline_top5_accs) if non_baseline_top5_accs else 0
    
    # 3. Weighted ASA Score (加权得分：Top-5 均值 * 一致性)
    weighted_score = overall_avg_top5_acc * consistency_score
    
    # 4. Style Resilience (风格韧性) - 使用 Top-5 计算
    style_resilience = 1.0
    if "standard_1girl" in summary and "complex_background" in summary:
        s1g = summary["standard_1girl"]["top5_accuracy"]
        cbg = summary["complex_background"]["top5_accuracy"]
        style_resilience = cbg / s1g if s1g > 0 else 0
    elif len(summary) == 1:
        style_resilience = 1.0 # 单一类型无从谈起韧性，设为基准 1.0

    # 5. Delta vs Baseline - 使用 Top-5 计算
    delta_vs_baseline = 0
    if baseline_pt and baseline_pt in summary:
        # 如果有非 baseline 数据，计算增益；否则增益为 0
        other_accs = [m['top5_accuracy'] for pt, m in summary.items() if pt != baseline_pt]
        if other_accs:
            delta_vs_baseline = np.mean(other_accs) - summary[baseline_pt]['top5_accuracy']

    metrics_final = {
        "overall_avg_acc": overall_avg_top5_acc, # 此时该值代表 Top-5
        "consistency_score": consistency_score,
        "weighted_score": weighted_score,
        "style_resilience": style_resilience,
        "delta_vs_baseline": delta_vs_baseline
    }

    final_output = {
        "benchmark_summary": summary,
        "metrics": metrics_final,
        "config_snapshot": config
    }
    
    if out_cfg['save_detailed_json']:
        final_output["details"] = detailed_results

    # 保存报告
    report_path = report_dir / "asa_benchmark_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)

    # 生成可视化
    generate_visualization(summary, metrics_final, report_dir, baseline_pt)

    print(f"\n{'='*40}")
    print(f"ASA Benchmark Report: {report_path}")
    print(f"Visualization: {report_dir / 'asa_benchmark_chart.png'}")
    print(f"{'-'*40}")
    print(f"Weighted Score:   {weighted_score:.4f} (Top-5 Focus)")
    print(f"Overall Avg Top-5: {overall_avg_top5_acc:.2%}")
    print(f"Consistency Score: {consistency_score:.4f}")
    print(f"Style Resilience:  {style_resilience:.4f}")
    print(f"Delta vs Baseline: {delta_vs_baseline:+.4f}")
    print(f"{'='*40}")

if __name__ == "__main__":
    print("Script started...")
    try:
        main()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
