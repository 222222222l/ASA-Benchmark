import requests
import json
import os
import time
import re
from pathlib import Path
from tqdm import tqdm

# 加载全局配置
def load_config():
    config_path = Path(__file__).parent / "benchmark_config.json"
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()
EVAL_CFG = CONFIG['evaluation_settings']
DATA_CFG = CONFIG['test_data_settings']

COMFYUI_URL = EVAL_CFG['comfyui_api_url']
CHECKPOINT = EVAL_CFG['target_checkpoint']
ARTISTS_CSV = DATA_CFG['test_artists_csv']
OUTPUT_ROOT = Path(DATA_CFG['generated_images_root'])
BASELINE_PROMPT = DATA_CFG.get('baseline_prompt', "1girl, solo, long hair, looking at viewer") # 默认保底

def get_workflow(prompt, artist_name, checkpoint, safe_filename):
    full_prompt = f"artist:{artist_name}, {prompt}" if artist_name else prompt
    return {
        "3": {"inputs": {"seed": 42, "steps": 20, "cfg": 7, "sampler_name": "euler", "scheduler": "normal", "denoise": 1, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0], "latent_image": ["5", 0]}, "class_type": "KSampler"},
        "4": {"inputs": {"ckpt_name": checkpoint}, "class_type": "CheckpointLoaderSimple"},
        "5": {"inputs": {"width": 1024, "height": 1024, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "6": {"inputs": {"text": full_prompt, "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "7": {"inputs": {"text": "nsfw, low quality, bad anatomy", "clip": ["4", 1]}, "class_type": "CLIPTextEncode"},
        "8": {"inputs": {"samples": ["3", 0], "vae": ["4", 2]}, "class_type": "VAEDecode"},
        "9": {"inputs": {"filename_prefix": f"ASA_TEMP_{safe_filename}", "images": ["8", 0]}, "class_type": "SaveImage"}
    }

def queue_prompt(workflow):
    p = {"prompt": workflow}
    res = requests.post(f"{COMFYUI_URL}/prompt", data=json.dumps(p).encode('utf-8'))
    res.raise_for_status()
    return res.json()['prompt_id']

def get_history(prompt_id):
    res = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
    res.raise_for_status()
    return res.json().get(prompt_id)

def download_image(filename, subfolder, type, save_path):
    params = {"filename": filename, "subfolder": subfolder, "type": type}
    res = requests.get(f"{COMFYUI_URL}/view", params=params)
    res.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(res.content)

def main():
    # 1. 准备任务列表 (Prompt Types x Artists)
    prompt_types = DATA_CFG['prompt_types']
    prompt_configs = DATA_CFG.get('prompt_configs', {})
    
    # 加载画师列表
    with open(ARTISTS_CSV, 'r', encoding='utf-8') as f:
        artists_base = [line.strip() for line in f if line.strip()]

    # 构建总任务列表: (prompt_type, artist_name)
    all_tasks = []
    for pt in prompt_types:
        # 每个目录下先加一个 Baseline
        all_tasks.append((pt, None))
        for artist in artists_base:
            all_tasks.append((pt, artist))

    print(f"Total tasks to queue: {len(all_tasks)} ({len(prompt_types)} types x {len(artists_base)+1} artists)")

    # 2. 任务执行循环
    pending_tasks = {} # prompt_id -> (prompt_type, safe_name)
    pbar = tqdm(total=len(all_tasks), desc="Processing")
    
    task_idx = 0
    while task_idx < len(all_tasks) or pending_tasks:
        # A. 填充队列 (保持 ComfyUI 队列中有任务，不超过 10 个)
        while task_idx < len(all_tasks) and len(pending_tasks) < 10:
            pt, artist = all_tasks[task_idx]
            prompt_str = prompt_configs.get(pt, BASELINE_PROMPT)
            
            safe_name = re.sub(r'[^\w\s\(\)\-]', '', artist).strip() if artist else "GLOBAL_BASELINE"
            workflow = get_workflow(prompt_str, artist, CHECKPOINT, safe_name)
            
            try:
                pid = queue_prompt(workflow)
                pending_tasks[pid] = (pt, safe_name)
                task_idx += 1
            except Exception as e:
                print(f"Error queuing task: {e}")
                time.sleep(5)
        
        # B. 轮询结果并下载
        completed_ids = []
        for pid, (pt, name) in pending_tasks.items():
            try:
                history = get_history(pid)
                if history and 'outputs' in history:
                    # 获取生成的图片信息
                    images = history['outputs'].get('9', {}).get('images', [])
                    save_dir = OUTPUT_ROOT / pt
                    save_dir.mkdir(parents=True, exist_ok=True)
                    
                    for img_info in images:
                        save_path = save_dir / f"ASA_Result_{name}_{pid[:8]}.png"
                        download_image(img_info['filename'], img_info['subfolder'], img_info['type'], save_path)
                    
                    completed_ids.append(pid)
                    pbar.update(1)
            except Exception as e:
                print(f"Error polling history for {pid}: {e}")
                time.sleep(1)
        
        for pid in completed_ids:
            del pending_tasks[pid]
            
        if pending_tasks:
            time.sleep(2)

    pbar.close()
    print(f"\nDone! All images saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()
