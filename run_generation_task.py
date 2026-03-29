import requests
import json
import os
import time
from pathlib import Path
from tqdm import tqdm

# --- 配置区 ---
COMFYUI_API_URL = "http://127.0.0.1:8180"
CHECKPOINT_NAME = "chenkin-0.39.safetensors"
ARTISTS_LIST_PATH = r"e:\artist-benchmark\test_artists_list.csv"
OUTPUT_ROOT = r"e:\artist-benchmark\test_generated_outputs"
BASELINE_PROMPT = "1girl, solo, long hair, breasts, looking at viewer, simple background, blonde hair, long sleeves, white background, gloves, hat, closed mouth, very long hair, grey hair, ass, cowboy shot, small breasts, puffy sleeves, pink eyes, from side, puffy short sleeves, black headwear, expressionless, half-closed eyes, juliet sleeves, monster girl, cropped legs, arms at sides, arched back, jester cap, puff and slash sleeves, black gloves, elbow gloves, short sleeves, looking to the side"

# 简化的 ComfyUI API 工作流 JSON (API 格式)
# 注意：这需要您 ComfyUI 中有对应的 Load Checkpoint 和 KSampler 节点 ID
# 以下是一个典型的 API 结构模板，您可能需要根据 ComfyUI 的 'Save API Format' 进行微调
import re

def get_workflow(prompt, artist_name, checkpoint, safe_filename):
    # 组合最终 Prompt (使用带转义的名称)
    full_prompt = f"artist:{artist_name}, {prompt}" if artist_name else prompt
    
    workflow = {
        "3": {
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 5,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler"
        },
        "4": {
            "inputs": {
                "ckpt_name": checkpoint
            },
            "class_type": "CheckpointLoaderSimple"
        },
        "5": {
            "inputs": {
                "width": 832,
                "height": 1216,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage"
        },
        "6": {
            "inputs": {
                "text": full_prompt,
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "7": {
            "inputs": {
                "text": "nsfw, low quality, bad anatomy",
                "clip": ["4", 1]
            },
            "class_type": "CLIPTextEncode"
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2]
            },
            "class_type": "VAEDecode"
        },
        "9": {
            "inputs": {
                # 扁平化命名，完全避免触发 ComfyUI 后端的路径扫描 Bug
                "filename_prefix": f"ASA_Result_zero_artist_{safe_filename}",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }
    return workflow

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    try:
        response = requests.post(f"{COMFYUI_API_URL}/prompt", data=data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        if hasattr(e, 'response') and e.response is not None:
            print(f"\n[Error Detail] {e.response.text}")
        raise RuntimeError(f"Failed to queue prompt: {e}")

def main():
    # --- 环境自愈补丁 ---
    # 针对 ComfyUI (Aki版) 校验 Bug：自动创建缺失的 output 内部路径
    bug_fix_path = Path(r"F:\ComfyUI-aki-v1.3\output\checkpoints")
    if not bug_fix_path.exists():
        print(f"Applying environment patch: creating {bug_fix_path}")
        bug_fix_path.mkdir(parents=True, exist_ok=True)

    # 1. 加载艺术家列表
    if not os.path.exists(ARTISTS_LIST_PATH):
        print(f"Error: {ARTISTS_LIST_PATH} not found.")
        return

    with open(ARTISTS_LIST_PATH, 'r', encoding='utf-8') as f:
        artists = [line.strip() for line in f if line.strip()]

    if not artists:
        print("Error: No artists found in the list.")
        return

    # 2. 生成图像任务
    print(f"Starting image generation for {len(artists)} artists...")
    
    # 创建根目录
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)

    success_count = 0
    for artist in tqdm(artists, desc="Queuing Tasks"):
        # 更加彻底的文件名清洗逻辑
        # 仅保留中英文字符、数字、空格、括号、短横线
        # 将任何可能被误认为路径的字符（如 . \ / : 等）全部移除
        safe_artist_name = re.sub(r'[^\w\s\(\)\-]', '', artist).strip()
        # 针对 Windows 特殊文件夹名限制进行额外安全处理
        if not safe_artist_name or safe_artist_name.upper() in ["CON", "PRN", "AUX", "NUL", "COM1", "LPT1"]:
            safe_artist_name = f"artist_{hash(artist)}"
        
        # 提示词传原名 (带转义)，文件名传清洗后的安全名
        workflow = get_workflow(BASELINE_PROMPT, artist, CHECKPOINT_NAME, safe_artist_name)
        
        try:
            queue_prompt(workflow)
            success_count += 1
        except Exception as e:
            print(f"\n[Error] Stopped at artist '{artist}': {e}")
            break

    print(f"\n[Success] {success_count} tasks queued successfully.")
    print(f"\n[Notice] 任务已全部下发至 ComfyUI 队列。")
    print(f"请检查 ComfyUI 的 output 目录，并将生成的图像按以下结构整理至 {OUTPUT_ROOT}：")
    print(f"  {OUTPUT_ROOT}/zero_artist/[艺术家名]/*.png")
    print("\n提示：ComfyUI 默认将图像保存在其安装目录下的 output 文件夹中。")

if __name__ == "__main__":
    main()
