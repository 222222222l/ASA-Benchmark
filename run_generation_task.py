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
def get_workflow(prompt, artist_name, checkpoint):
    # 组合最终 Prompt
    full_prompt = f"artist:{artist_name}, {prompt}" if artist_name else prompt
    
    workflow = {
        "3": {
            "inputs": {
                "seed": 42,
                "steps": 20,
                "cfg": 7,
                "sampler_name": "euler",
                "scheduler": "normal",
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
                "width": 1024,
                "height": 1024,
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
                "filename_prefix": f"ASA_Test_{artist_name}",
                "images": ["8", 0]
            },
            "class_type": "SaveImage"
        }
    }
    return workflow

def queue_prompt(prompt_workflow):
    p = {"prompt": prompt_workflow}
    data = json.dumps(p).encode('utf-8')
    response = requests.post(f"{COMFYUI_API_URL}/prompt", data=data)
    return response.json()

def main():
    # 1. 加载艺术家列表
    with open(ARTISTS_LIST_PATH, 'r', encoding='utf-8') as f:
        artists = [line.strip() for line in f if line.strip()]

    # 2. 生成图像任务
    # 为了简化，我们只演示 zero_artist (Baseline) 逻辑
    # 实际执行时，需要确保 ComfyUI 正在运行
    print(f"Starting image generation for {len(artists)} artists...")
    
    # 创建 Baseline 目录
    baseline_dir = Path(OUTPUT_ROOT) / "zero_artist"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    for artist in tqdm(artists):
        # 注意：这里仅作为逻辑演示。ComfyUI API 调用需要处理队列等待和文件下载。
        # 在实际环境中，您可能需要更复杂的脚本来从 API 获取生成的图像并保存到指定路径。
        # print(f"Queuing: {artist}")
        # workflow = get_workflow(BASELINE_PROMPT, artist, CHECKPOINT_NAME)
        # queue_prompt(workflow)
        pass

    print("\n[Notice] 图像生成脚本已准备就绪。")
    print("由于 API 环境依赖 ComfyUI 运行且工作流节点 ID 可能不同，请确保：")
    print(f"1. ComfyUI 正在 {COMFYUI_API_URL} 运行。")
    print(f"2. 您的工作流中 CheckpointLoaderSimple 节点 ID 为 4，KSampler 为 3。")
    print("3. 生成的图像已放置在 e:\\artist-benchmark\\test_generated_outputs\\zero_artist 目录下。")

if __name__ == "__main__":
    main()
