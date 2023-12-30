import os
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from torchvision.utils import save_image

import argparse

from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline
from DeepCache.sdxl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline as DeepCacheStableDiffusionXLPipeline
from DeepCache.sdxl.pipeline_stable_diffusion_xl_img2img import StableDiffusionXLImg2ImgPipeline as DeepCacheStableDiffusionXLImg2ImgPipeline

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default = "stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--refine", action="store_true")
    parser.add_argument("--refine_model", type=str, default = "stabilityai/stable-diffusion-xl-refiner-1.0")

    parser.add_argument("--prompt", type=str, default='a photo of an astronaut on a moon')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    prompt = args.prompt
   
    baseline_pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda:0")
    if args.refine:
        refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            args.refine_model, 
            text_encoder_2=baseline_pipe.text_encoder_2,
            vae=baseline_pipe.vae,
            torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        ).to("cuda:0")
    
    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(0):
        set_random_seed(seed)
        _ = baseline_pipe(prompt, output_type='pt').images
    
    # Baseline
    logging.info("Running baseline...")
    start_time = time.time()
    set_random_seed(seed)

    if args.refine:
        image = baseline_pipe(
            prompt=prompt,
            num_inference_steps=50,
            denoising_end=0.8,
            output_type="latent",
        ).images
        base_use_time = time.time() - start_time
        logging.info("Baseline - Base: {:.2f} seconds".format(base_use_time))
        start_time = time.time()
        ori_image = refiner_pipe(
            prompt=prompt,
            num_inference_steps=50,
            denoising_start=0.8,
            image=image,
            output_type="pt"
        ).images
        refine_use_time = time.time() - start_time
        logging.info("Baseline - Refiner: {:.2f} seconds".format(refine_use_time))
        baseline_use_time = base_use_time + refine_use_time
    else:
        ori_image = baseline_pipe(prompt, num_inference_steps=50, output_type='pt').images
        baseline_use_time = time.time() - start_time
        logging.info("Baseline: {:.2f} seconds".format(baseline_use_time))

    del baseline_pipe
    torch.cuda.empty_cache()
    
    # DeepCache
    pipe = DeepCacheStableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda:0")
    if args.refine:
        refiner_pipe = DeepCacheStableDiffusionXLImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        ).to("cuda:0")

    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(1):
        set_random_seed(seed)
        _ = pipe(
            prompt, 
            cache_interval=3, cache_layer_id=0, cache_block_id=0,
            output_type='pt', return_dict=True,
        ).images

    logging.info("Running DeepCache...")
    set_random_seed(seed)
    start_time = time.time()
    if args.refine:
        deepcache_base_output = pipe(
            prompt, 
            num_inference_steps=50,
            denoising_end=0.8, output_type="latent",
            cache_interval=3, cache_layer_id=0, cache_block_id=0,
            uniform=True,  
            return_dict=True
        ).images
        base_use_time = time.time() - start_time
        logging.info("DeepCache - Base: {:.2f} seconds".format(base_use_time))
        start_time = time.time()
        deepcache_output = refiner_pipe(
            prompt=prompt,
            num_inference_steps=50,
            denoising_start=0.8,
            cache_interval=3, cache_layer_id=0, cache_block_id=0,
            uniform=True, 
            image=deepcache_base_output,
            output_type='pt',
        ).images
        refine_use_time = time.time() - start_time
        logging.info("DeepCache - Refiner: {:.2f} seconds".format(refine_use_time))
        use_time = base_use_time + refine_use_time
    else:
        deepcache_output = pipe(
            prompt, 
            num_inference_steps=50,
            cache_interval=3, cache_layer_id=0, cache_block_id=0,
            uniform=True, 
            output_type='pt', 
            return_dict=True
        ).images
        use_time = time.time() - start_time
        logging.info("DeepCache: {:.2f} seconds".format(use_time))

    logging.info("Baseline: {:.2f} seconds. DeepCache: {:.2f} seconds".format(baseline_use_time, use_time))
    save_image([ori_image[0], deepcache_output[0]], "output.png")
    logging.info("Saved to output.png. Done!")



