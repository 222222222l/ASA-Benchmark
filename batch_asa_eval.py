import os
import sys
import json
import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from timm.models import create_model

# 确保能加载项目中的 lsnet 模型组件
# 将 comfyui-lsnet 加入 Python 路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir / "comfyui-lsnet"))

try:
    from lsnet_model import lsnet_artist  # noqa: F401
except ImportError:
    print("Error: Could not find lsnet_model in comfyui-lsnet directory. Please check the path.")
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
    """加载 class_mapping.csv，返回名称到 ID 的双向映射"""
    name_to_id = {}
    id_to_name = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row['class_name']
            sanitized_name = sanitize_name(raw_name, replace_underscore)
            class_id = int(row['class_id'])
            name_to_id[sanitized_name] = class_id
            id_to_name[class_id] = sanitized_name
    return name_to_id, id_to_name

def load_test_artists(csv_path: str, replace_underscore: bool) -> List[str]:
    """加载待测试的艺术家列表"""
    artists = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        # 假设 csv 只有一列或者第一列是艺术家名
        reader = csv.reader(f)
        # 跳过表头如果存在
        try:
            for row in reader:
                if row:
                    artists.append(sanitize_name(row[0].strip(), replace_underscore))
        except StopIteration:
            pass
    return artists

def get_asa_model(checkpoint_path: str, num_classes: int, device: str):
    """加载评测模型"""
    model = create_model('lsnet_xl_artist_448', pretrained=False, num_classes=num_classes)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    # 归一化权重键名
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
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
        f"Core Metrics Summary\n"
        f"{'-'*30}\n"
        f"Overall Avg Top-1: {metrics['overall_avg_acc']:.2%}\n"
        f"Consistency Score: {metrics['consistency_score']:.4f}\n"
        f"Delta vs Baseline: {metrics['delta_vs_baseline']:+.4f}\n"
        f"Style Resilience:  {metrics['style_resilience']:.4f}\n"
        f"{'-'*30}\n"
        f"Weighted Score:   {metrics['weighted_score']:.4f}\n"
        f"(Weighted Score = Avg_Acc * Consistency)"
    )
    ax2.text(0.1, 0.5, metrics_text, fontsize=12, family='monospace', va='center')

    fig.tight_layout()
    plt.savefig(report_dir / "asa_benchmark_chart.png")
    plt.close()

def main():
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

    # 2. 图像预处理
    transform = transforms.Compose([
        transforms.Resize(448),
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
            # 兼容下划线和空格的文件夹名，同时处理反转义
            base_name = unescape_name(artist_name)
            artist_dir = pt_dir / base_name
            if not artist_dir.exists():
                # 尝试下划线格式
                alt_name = base_name.replace(' ', '_')
                artist_dir = pt_dir / alt_name
            
            if not artist_dir.exists() or not artist_name in name_to_id:
                continue

            expected_id = name_to_id[artist_name]
            image_paths = [p for p in artist_dir.glob('*.*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']]
            
            if not image_paths:
                continue

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
                    is_top1 = (top5_indices[idx_in_batch][0] == expected_id).item()
                    is_top5 = (expected_id in top5_indices[idx_in_batch]).item()
                    conf = float(top5_probs[idx_in_batch][0])

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

    # --- 核心指标计算优化 ---
    # 1. Consistency Score (1 - std)
    consistency_score = 1.0 - np.std(non_baseline_accs) if len(non_baseline_accs) > 1 else 1.0
    
    # 2. Overall Avg Acc
    overall_avg_acc = np.mean(non_baseline_accs) if non_baseline_accs else 0
    
    # 3. Weighted ASA Score (加权得分：均值 * 一致性) - 惩罚“低准确率的高一致性”模型
    weighted_score = overall_avg_acc * consistency_score
    
    # 4. Style Resilience (风格韧性)
    # 定义为：复杂场景(complex_background) 相对于 简单场景(standard_1girl) 的准确率保留率
    style_resilience = 1.0
    if "standard_1girl" in summary and "complex_background" in summary:
        s1g = summary["standard_1girl"]["top1_accuracy"]
        cbg = summary["complex_background"]["top1_accuracy"]
        style_resilience = cbg / s1g if s1g > 0 else 0

    # 5. Delta vs Baseline
    delta_vs_baseline = 0
    if baseline_pt and baseline_pt in summary and len(non_baseline_accs) > 0:
        delta_vs_baseline = overall_avg_acc - summary[baseline_pt]['top1_accuracy']

    metrics_final = {
        "overall_avg_acc": overall_avg_acc,
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
    print(f"Weighted Score:   {weighted_score:.4f} (Primary Rank)")
    print(f"Overall Avg Acc:  {overall_avg_acc:.2%}")
    print(f"Consistency Score: {consistency_score:.4f}")
    print(f"Style Resilience:  {style_resilience:.4f}")
    print(f"Delta vs Baseline: {delta_vs_baseline:+.4f}")
    print(f"{'='*40}")

if __name__ == "__main__":
    main()
