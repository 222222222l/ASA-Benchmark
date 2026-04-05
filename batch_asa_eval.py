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
    # 使用 insert(0, ...) 确保优先搜索本地路径，解决 Linux 环境下的路径冲突
    sys.path.insert(0, lsnet_path)
    # 同时也将当前目录加入，以便兼容不同的导入习惯
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

print(f"Current dir: {current_dir}")
print(f"LSNet path: {lsnet_path}")
print(f"Python path (first 3): {sys.path[:3]}")

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
    import timm
    from timm.models import create_model
    
    # 额外检查 triton，这是 ska.py 的硬依赖，但在 Linux 下可能未安装
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
    except ImportError:
        print("WARNING: triton not found. SKA operators in LSNet may fail on GPU.")
        print("Please install triton: pip install triton")

    print("Loading lsnet_model...")
    # 尝试多种导入方式以增强鲁棒性
    try:
        from lsnet_model import lsnet_artist
    except ImportError as e:
        print(f"DEBUG: lsnet_model import failed: {e}")
        # 如果还是失败，打印更详细的路径信息
        if os.path.exists(lsnet_path):
            print(f"DEBUG: os.listdir(lsnet_path): {os.listdir(lsnet_path)}")
        else:
            print(f"DEBUG: LSNet path {lsnet_path} does not exist!")
        raise
    
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
    
    # 处理 PyTorch 2.6+ 默认 weights_only=True 导致的加载失败问题
    try:
        # 尝试显式设置 weights_only=False 以兼容包含 Namespace 等非权重对象的 checkpoint
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except TypeError:
        # 兼容不支持 weights_only 参数的老版本 PyTorch
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in state_dict:
        state_dict = state_dict['model']
    
    # 归一化权重键名
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # 容错加载：支持 Kaloscope 2.0 可能存在的 projection 层差异
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    return model

def generate_visualization(summary: dict, metrics: dict, report_dir: Path, baseline_pt: str, stats_per_artist: dict):
    """生成 Benchmark 可视化图表，包含四象限分析"""
    # 修复 Linux 环境下可能缺失 Arial 字体的警告，优先尝试 DejaVu Sans (Linux 常用)
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Bitstream Vera Sans', 'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 'Lucid', 'Arial', 'Helvetica', 'Avant Garde', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 增加子图数量，包含四象限图
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :]) # 底部通栏用于四象限分析
    
    # 子图 1: 各 Prompt Type 准确率
    prompt_types = list(summary.keys())
    top1_accs = [summary[pt]['top1_accuracy'] * 100 for pt in prompt_types]
    top5_accs = [summary[pt]['top5_accuracy'] * 100 for pt in prompt_types]

    x = np.arange(len(prompt_types))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, top1_accs, width, label='Top-1 Accuracy (%)', color='#3498db')
    rects2 = ax1.bar(x + width/2, top5_accs, width, label='Top-5 Accuracy (%)', color='#2ecc71')

    if baseline_pt and baseline_pt in summary:
        baseline_val = summary[baseline_pt]['top5_accuracy'] * 100
        ax1.axhline(y=baseline_val, color='r', linestyle='--', alpha=0.5, label=f'Baseline ({baseline_pt})')

    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('ASA Performance by Prompt Type')
    ax1.set_xticks(x)
    ax1.set_xticklabels(prompt_types, rotation=15)
    ax1.set_ylim(0, 105)
    ax1.legend()

    # 子图 2: 核心指标汇总文本
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
    ax2.text(0.1, 0.5, metrics_text, fontsize=14, family='monospace', va='center')

    # 子图 3: 画师级别四象限分析 (Top-5 Acc vs Confidence)
    artist_names = []
    artist_accs = []
    artist_confs = []
    
    for name, stats in stats_per_artist.items():
        if stats['total'] > 0:
            artist_names.append(name)
            artist_accs.append(stats['top5_hits'] / stats['total'] * 100)
            artist_confs.append(stats['conf_sum'] / stats['total'])
    
    if artist_accs:
        # 绘制散点
        scatter = ax3.scatter(artist_accs, artist_confs, alpha=0.6, c=artist_accs, cmap='viridis', s=50)
        
        # 绘制四象限分割线
        mean_acc = np.mean(artist_accs)
        mean_conf = np.mean(artist_confs)
        ax3.axvline(x=mean_acc, color='gray', linestyle='--', alpha=0.3)
        ax3.axhline(y=mean_conf, color='gray', linestyle='--', alpha=0.3)
        
        # 象限标注
        ax3.text(mean_acc + 2, 0.95, "Consistent & Accurate", fontsize=10, color='green', fontweight='bold')
        ax3.text(2, 0.95, "Confident but Wrong", fontsize=10, color='orange', fontweight='bold')
        ax3.text(2, 0.05, "Struggling", fontsize=10, color='red', fontweight='bold')
        
        ax3.set_xlabel('Artist Top-5 Accuracy (%)')
        ax3.set_ylabel('Avg Prediction Confidence')
        ax3.set_title('Artist-level Style Adherence (Four Quadrant Analysis)')
        ax3.set_xlim(-5, 105)
        ax3.set_ylim(0, 1.1)
        
        # 为极端值添加少量标注（前 5 名和后 5 名）
        sorted_idx = np.argsort(artist_accs)
        for i in list(sorted_idx[:3]) + list(sorted_idx[-3:]):
            ax3.annotate(artist_names[i], (artist_accs[i], artist_confs[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

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
    
    stats_per_prompt = {pt: {"correct": 0, "total": 0, "top5": 0, "conf_sum": 0, "baseline_acc": 0} for pt in prompt_types}
    stats_per_artist = {} # 用于四象限分析: {artist_name: {"top5_acc": [], "conf": []}}
    detailed_results = []

    for pt in prompt_types:
        print(f"\nEvaluating Prompt Type: {pt}")
        pt_dir = image_root / pt
        if not pt_dir.exists():
            print(f"[Warning] Prompt directory {pt_dir} not found. Skipping.")
            continue

        # --- 新增：处理 GLOBAL_BASELINE 图像 ---
        baseline_images = list(pt_dir.glob("*GLOBAL_BASELINE*.png"))
        if baseline_images:
            print(f"  Found {len(baseline_images)} Global Baseline images. Calculating chance level...")
            baseline_top5_hits = 0
            
            # 加载所有待测画师的 ID
            target_ids = []
            for artist_name in test_artists:
                mapping_key = get_mapping_key(artist_name)
                if mapping_key in name_to_id:
                    target_ids.append(name_to_id[mapping_key])
            
            if target_ids:
                for b_path in baseline_images:
                    try:
                        b_img = transform(Image.open(b_path).convert('RGB')).unsqueeze(0).to(eval_cfg['device'])
                        with torch.no_grad():
                            b_logits = model(b_img)
                            b_probs = F.softmax(b_logits, dim=-1)
                            _, b_top5_indices = torch.topk(b_probs, k=5, dim=-1)
                            b_pred_ids = b_top5_indices[0].tolist()
                            
                            # 计算在该 baseline 图像中，有多少个目标画师被“误中”了
                            # 这种误中率反映了模型的固有偏好（Chance Level）
                            hits = sum(1 for tid in target_ids if tid in b_pred_ids)
                            baseline_top5_hits += (hits / len(target_ids))
                    except Exception as e:
                        print(f"  Error processing baseline image {b_path.name}: {e}")
                
                stats_per_prompt[pt]["baseline_acc"] = baseline_top5_hits / len(baseline_images)
                print(f"  Global Baseline Top-5 Accuracy (Chance Level): {stats_per_prompt[pt]['baseline_acc']:.4%}")

        for artist_name in tqdm(test_artists, desc=f"Artists in {pt}"):
            if artist_name not in stats_per_artist:
                stats_per_artist[artist_name] = {"top5_hits": 0, "total": 0, "conf_sum": 0}
            
            # 1. 获取映射表中的键名 (原始下划线格式)
            mapping_key = get_mapping_key(artist_name)
            
            # 2. 清洗出 safe_name (对应 run_generation_task.py 中的逻辑，用于文件名匹配)
            safe_name = re.sub(r'[^\w\s\(\)\-]', '', artist_name).strip()
            
            # 3. 在目录中寻找匹配该艺术家 safe_name 的所有文件
            pattern = f"*{safe_name}*.png"
            image_paths = list(pt_dir.glob(pattern))
            
            # 如果没找到，尝试原始 unescaped 名称
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
                    print(f"Error loading images in {artist_name}: {e}")
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

                    stats_per_prompt[pt]["total"] += 1
                    if is_top1: stats_per_prompt[pt]["correct"] += 1
                    if is_top5: stats_per_prompt[pt]["top5"] += 1
                    stats_per_prompt[pt]["conf_sum"] += conf
                    
                    # 记录画师级别统计
                    stats_per_artist[artist_name]["total"] += 1
                    if is_top5: stats_per_artist[artist_name]["top5_hits"] += 1
                    stats_per_artist[artist_name]["conf_sum"] += conf

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
                "sample_count": stats["total"],
                "baseline_chance_level": stats.get("baseline_acc", 0) # 记录该 Prompt Type 下的 Chance Level
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
    # 优先使用 GLOBAL_BASELINE 图像算出的 Chance Level 作为基准
    # 如果没找到 GLOBAL_BASELINE，则回退到对比不同 Prompt Type 的逻辑
    delta_vs_baseline = 0
    avg_chance_level = np.mean([m['baseline_chance_level'] for m in summary.values() if m['baseline_chance_level'] > 0])
    
    if avg_chance_level > 0:
        delta_vs_baseline = overall_avg_top5_acc - avg_chance_level
        print(f"  Using Global Baseline Chance Level ({avg_chance_level:.4%}) for Delta calculation.")
    elif baseline_pt and baseline_pt in summary:
        other_accs = [m['top5_accuracy'] for pt, m in summary.items() if pt != baseline_pt]
        if other_accs:
            delta_vs_baseline = np.mean(other_accs) - summary[baseline_pt]['top5_accuracy']
            print(f"  Using Prompt Type '{baseline_pt}' as reference for Delta calculation.")

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
    generate_visualization(summary, metrics_final, report_dir, baseline_pt, stats_per_artist)

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
