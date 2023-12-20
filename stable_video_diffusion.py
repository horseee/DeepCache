import time
import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
from DeepCache.svd.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline as DeepCacheStableVideoDiffusionPipeline

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
file_name = "rocket"
image = image.resize((1024, 576))

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

generator = torch.manual_seed(42)
logging.info("Running baseline...")
start_time = time.time()
frames = pipe(
    image, 
    decode_chunk_size=8, generator=generator,
).frames[0]
origin_use_time = time.time() - start_time

export_to_video(frames, "{}_origin.mp4".format(file_name), fps=7)
logging.info("Origin: {:.2f} seconds".format(origin_use_time))

del pipe
torch.cuda.empty_cache()

deepcache_pipe = DeepCacheStableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16",
)
deepcache_pipe.enable_model_cpu_offload()

generator = torch.manual_seed(42)
logging.info("Running DeepCache...")
start_time = time.time()
frames = deepcache_pipe(
    image, 
    decode_chunk_size=8, generator=generator,
    cache_interval=3, cache_branch=0,
).frames[0]
deepcache_use_time = time.time() - start_time

export_to_video(frames, "{}_deepcache.mp4".format(file_name), fps=7)
logging.info("DeepCache: {:.2f} seconds".format(deepcache_use_time))
