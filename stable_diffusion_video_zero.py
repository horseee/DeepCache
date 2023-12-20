import os
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from torchvision.utils import save_image

import argparse

from DeepCache import StableDiffusionPipeline as DeepCacheStableDiffusionPipeline
from DeepCache import TextToVideoZeroPipeline as DeepCacheTextToVideoZeroPipeline
from diffusers import StableDiffusionPipeline

from diffusers import TextToVideoZeroPipeline
from diffusers import TextToVideoSDPipeline
from diffusers.utils import export_to_video

import shutil
import imageio

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='runwayml/stable-diffusion-v1-5')#model_id_v2_1 = 'stabilityai/stable-diffusion-2-1'
    parser.add_argument("--prompt", type=str, default='An astronaut riding a horse.')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed = args.seed
    prompt = args.prompt


    # baseline_pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda:0")
    # baseline_pipe = TextToVideoZeroPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    # baseline_pipe = TextToVideoSDPipeline.from_pretrained(args.model, torch_dtype=torch.float16, variant="fp16")

    baseline_pipe = TextToVideoZeroPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    # baseline_pipe.enable_model_cpu_offload()
    for _ in range(2):
        set_random_seed(seed)
        _ = baseline_pipe(prompt).images
        
    # Baseline
    logging.info("Running baseline...")
    
    set_random_seed(seed)
    # baseline_pipe.enable_model_cpu_offload()
    # ori_output = baseline_pipe(prompt).frames
    prompt = "a high quality realistic photo of a panda surfing on a wakeboard"
    start_time = time.time()
    ori_output = baseline_pipe(prompt=prompt).images
    use_time = time.time() - start_time
    ori_output = [(r * 255).astype("uint8") for r in ori_output]
    imageio.mimsave("ori_video.mp4", ori_output, fps=4)
    logging.info("Baseline: {:.2f} seconds".format(use_time))
    
    # video_path = export_to_video(ori_output, output_video_path="./video_ori.mp4", fps=24)
    # print(video_path)
    # shutil.copy(video_path, 'video.mp4')
    # imageio.mimsave("video.mp4", video_path, fps=4)
    # save_image(image_ori[0], "{}_{:.2f}.png".format(prompt, use_time))
    del baseline_pipe
    torch.cuda.empty_cache()

    # DeepCache
    pipe = DeepCacheTextToVideoZeroPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(2):
        set_random_seed(seed)
        _ = pipe(prompt).images

    logging.info("Running DeepCache...")
    set_random_seed(seed)
    start_time = time.time()
    deepcache_output = pipe(
        prompt, 
        cache_interval=5, cache_layer_id=0, cache_block_id=0,
        uniform=False, pow=1.4, center=15,
    ).images

    use_time = time.time() - start_time
    deepcache_output = [(r * 255).astype("uint8") for r in deepcache_output]
    imageio.mimsave("dc_video.mp4", deepcache_output, fps=4)
    logging.info("DeepCache: {:.2f} seconds".format(use_time))

    # save_image([ori_output[0], deepcache_output[0]], "output.png")
    # logging.info("Saved to output.png. Done!")





