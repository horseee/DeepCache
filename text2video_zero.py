import os
import time
import torch
from torchvision.utils import save_image

import argparse

from DeepCache.sd.pipeline_text_to_video_zero import TextToVideoZeroPipeline as DeepCacheTextToVideoZeroPipeline
from diffusers import TextToVideoZeroPipeline

import imageio

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='runwayml/stable-diffusion-v1-5')#model_id_v2_1 = 'stabilityai/stable-diffusion-2-1'
    parser.add_argument("--prompt", type=str, default='A panda is playing guitar on times square')
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    prompt = args.prompt

    baseline_pipe = TextToVideoZeroPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    # baseline_pipe.enable_model_cpu_offload()
    for _ in range(1):
        set_random_seed(seed)
        _ = baseline_pipe(prompt).images
        
    # Baseline
    logging.info("Running baseline...")
    
    set_random_seed(seed)
    start_time = time.time()
    ori_output = baseline_pipe(prompt=prompt).images
    use_time = time.time() - start_time
    ori_output = [(r * 255).astype("uint8") for r in ori_output]
    imageio.mimsave("original_video.gif", ori_output, fps=4, loop=0)
    logging.info("Baseline: {:.2f} seconds".format(use_time))
    
    del baseline_pipe
    torch.cuda.empty_cache()

    # DeepCache
    pipe = DeepCacheTextToVideoZeroPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(1):
        set_random_seed(seed)
        _ = pipe(prompt).images

    logging.info("Running DeepCache...")
    set_random_seed(seed)
    start_time = time.time()
    deepcache_output = pipe(
        prompt, 
        cache_interval=3, cache_layer_id=0, cache_block_id=0,
        uniform=True,
    ).images
    
    use_time = time.time() - start_time
    deepcache_output = [(r * 255).astype("uint8") for r in deepcache_output]
    imageio.mimsave(f"deepcache_video.gif", deepcache_output, fps=4, loop=0)
    logging.info("DeepCache: {:.2f} seconds".format(use_time))




