import os
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from torchvision.utils import save_image

from DeepCache.pipeline_stable_diffusion import StableDiffusionPipeline

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    model_id_v2_1 = 'stabilityai/stable-diffusion-2-1'
    model_id_v1_5 = 'runwayml/stable-diffusion-v1-5'
    pipe = StableDiffusionPipeline.from_pretrained(model_id_v1_5, torch_dtype=torch.float16).to("cuda:0")

    seed = 42
    prompt = "a photo of an astronaut on a moon"

    # Warmup GPU
    logging.info("Warming up GPU...")
    for _ in range(2):
        set_random_seed(seed)
        image_ori = pipe(prompt, output_type='pt', output_all_sequence=False).images
        
    # Baseline
    logging.info("Running baseline...")
    start_time = time.time()
    set_random_seed(seed)
    ori_output = pipe(prompt, output_type='pt', output_all_sequence=False).images
    use_time = time.time() - start_time
    logging.info("Baseline: {:.2f} seconds".format(use_time))
    #save_image(image_ori[0], "{}_{:.2f}.png".format(prompt, use_time))

    
    # DeepCache
    logging.info("Running DeepCache...")
    set_random_seed(seed)
    start_time = time.time()
    deepcache_output = pipe(
        prompt, 
        update_feature_interval=5, 
        replicate_layer_number=0, replicate_block_number=0,
        uniform=True, pow=1.4, center=15,
        output_type='pt', return_dict=True
    ).images
    use_time = time.time() - start_time
    logging.info("DeepCache: {:.2f} seconds".format(use_time))

    save_image([ori_output[0], deepcache_output[0]], "output.png")
    logging.info("Saved to output.png. Done!")




