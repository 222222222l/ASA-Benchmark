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
    # 1. 准备目录
    test_type = DATA_CFG['prompt_types'][0] # 默认为第一个类型，如 my_test
    save_dir = OUTPUT_ROOT / test_type
    save_dir.mkdir(parents=True, exist_ok=True)

    # 2. 加载画师列表
    with open(ARTISTS_CSV, 'r', encoding='utf-8') as f:
        artists = [line.strip() for line in f if line.strip()]

    # 3. 任务队列循环
    pending_tasks = {} # prompt_id -> artist_name
    
    print(f"Starting Benchmark Generation: {len(artists)} artists...")
    
    # 插入 Baseline 任务
    print("Queuing Global Baseline...")
    artists.insert(0, None) 

    pbar = tqdm(total=len(artists), desc="Processing")
    
    artist_idx = 0
    while artist_idx < len(artists) or pending_tasks:
        # A. 填充队列 (保持 ComfyUI 队列中有任务，但不超过 10 个以防压力过大)
        while artist_idx < len(artists) and len(pending_tasks) < 10:
            artist = artists[artist_idx]
            safe_name = re.sub(r'[^\w\s\(\)\-]', '', artist).strip() if artist else "GLOBAL_BASELINE"
            workflow = get_workflow(BASELINE_PROMPT, artist, CHECKPOINT, safe_name)
            
            try:
                pid = queue_prompt(workflow)
                pending_tasks[pid] = safe_name
                artist_idx += 1
            except:
                time.sleep(5) # 出错等待
        
        # B. 轮询结果并下载
        completed_ids = []
        for pid, name in pending_tasks.items():
            history = get_history(pid)
            if history and 'outputs' in history:
                # 获取生成的图片信息
                images = history['outputs'].get('9', {}).get('images', [])
                for img_info in images:
                    save_path = save_dir / f"ASA_Result_{name}_{pid[:8]}.png"
                    download_image(img_info['filename'], img_info['subfolder'], img_info['type'], save_path)
                completed_ids.append(pid)
                pbar.update(1)
        
        for pid in completed_ids:
            del pending_tasks[pid]
            
        if pending_tasks:
            time.sleep(2) # 轮询间隔

    pbar.close()
    print(f"\nDone! All images saved directly to: {save_dir}")

if __name__ == "__main__":
    main()
