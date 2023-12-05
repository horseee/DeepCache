import os
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from torchvision.utils import save_image

import argparse

from diffusers import StableDiffusionXLPipeline
from DeepCache import StableDiffusionXLPipeline as DeepCacheStableDiffusionXLPipeline

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = "stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--prompt", type=str, default='a photo of an astronaut on a moon')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    prompt = args.prompt
    prompt = "A photo of an astronaut on a moon"


    baseline_pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda:0")

    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(2):
        set_random_seed(seed)
        _ = baseline_pipe(prompt, output_type='pt').images
        
    # Baseline
    logging.info("Running baseline...")
    start_time = time.time()
    set_random_seed(seed)
    ori_output = baseline_pipe(prompt, output_type='pt').images
    use_time = time.time() - start_time
    logging.info("Baseline: {:.2f} seconds".format(use_time))
    del baseline_pipe
    torch.cuda.empty_cache()

    # DeepCache
    pipe = DeepCacheStableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda:0")

    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(2):
        set_random_seed(seed)
        _ = pipe(
            prompt, 
            cache_interval=3, cache_layer_id=0, cache_block_id=0,
            output_type='pt', return_dict=True
        ).images

    logging.info("Running DeepCache...")
    set_random_seed(seed)
    start_time = time.time()
    deepcache_output = pipe(
        prompt, 
        cache_interval=3, cache_layer_id=0, cache_block_id=0,
        uniform=True, 
        output_type='pt', return_dict=True
    ).images
    use_time = time.time() - start_time
    logging.info("DeepCache: {:.2f} seconds".format(use_time))

    save_image([ori_output[0], deepcache_output[0]], "output.png")
    logging.info("Saved to output.png. Done!")



