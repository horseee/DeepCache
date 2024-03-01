
import time
import argparse
import numpy as np
import random

import os
from tqdm import tqdm

import torch
from datasets import load_dataset


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    if args.dataset == 'parti':
        prompts = load_dataset("nateraw/parti-prompts", split="train")
    elif args.dataset == 'coco2017':
        dataset = load_dataset("phiyodr/coco2017")
        prompts = [{"Prompt": sample['captions'][0]} for sample in dataset['validation']]
    else:
        raise NotImplementedError

    # Fixing these sample prompts in the interest of reproducibility.
    if args.original:
        from diffusers import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
    elif args.bk is not None:
        from diffusers import StableDiffusionPipeline
        if args.bk == "base":
            pipe = StableDiffusionPipeline.from_pretrained('nota-ai/bk-sdm-base-2m', torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
        elif args.bk == "small":
            pipe = StableDiffusionPipeline.from_pretrained('nota-ai/bk-sdm-small-2m', torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
        elif args.bk == "tiny":
            pipe = StableDiffusionPipeline.from_pretrained('nota-ai/bk-sdm-tiny-2m', torch_dtype=torch.float16, safety_checker=None).to("cuda:0")
    else:
        from DeepCache.sd.pipeline_stable_diffusion import StableDiffusionPipeline
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda:0")

    start_time = time.time()
    image_list, prompt_list = [], []
    num_batch = len(prompts) // args.batch_size 
    if len(prompts) % args.batch_size != 0:
        num_batch += 1

    for i in tqdm(range(num_batch)):
        start, end = args.batch_size * i, min(args.batch_size * (i + 1), len(prompts))
        sample_prompts = [prompts[i]["Prompt"] for i in range(start, end)] #[prompts[i] for i in range(start, end)]#
        set_random_seed(args.seed)
        if args.original or args.bk:
            pipe_output = pipe(
                sample_prompts, output_type='np', return_dict=True,
                num_inference_steps=args.steps
            )
        else:
            pipe_output = pipe(
                sample_prompts, num_inference_steps=args.steps,
                cache_interval=args.update_interval, 
                cache_layer_id=args.layer, cache_block_id=args.block, 
                uniform=args.uniform, pow=args.pow, center=args.center,
                output_type='np', return_dict=True
            )

        images = pipe_output.images
        images_int = (images * 255).astype("uint8")
        torch_int_images = torch.from_numpy(images_int).permute(0, 3, 1, 2)

        image_list.append(torch_int_images)
        prompt_list += sample_prompts

    use_time = round(time.time() - start_time, 2)
    all_images = torch.cat(image_list, dim=0)

    if not os.path.exists(f"{args.dataset}_ckpt"):
        os.makedirs(f"{args.dataset}_ckpt")

    if args.original:
        torch.save({
            "images": all_images,
            "prompts": prompt_list,
        }, f"{args.dataset}_ckpt/images-original-{args.steps}-time-{use_time}.pt")
    elif args.bk is not None:
        torch.save({
            "images": all_images,
            "prompts": prompt_list,
        }, f"{args.dataset}_ckpt/images-bksdm-{args.bk}-{args.steps}-time-{use_time}.pt")
    else:
        torch.save({
            "images": all_images,
            "prompts": prompt_list,
        }, f"{args.dataset}_ckpt/images-deepcache-{args.steps}-block-{args.block}-layer-{args.layer}-interval-{args.update_interval}-uniform-{args.uniform}-pow-{args.pow}-center-{args.center}-time-{use_time}.pt")
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)

    # For choosing baselines. If these two are not set, then it will use DeepCache.
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--bk", type=str, default=None)

    # Hyperparameters for DeepCache
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--update_interval", type=int, default=None)
    parser.add_argument("--uniform", action="store_true", default=False)
    parser.add_argument("--pow", type=float, default=None)
    parser.add_argument("--center", type=int, default=None)

    # Sampling setup
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_random_seed(args.seed)
    main(args)




