import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, StableVideoDiffusionPipeline
from torchvision.utils import save_image
import argparse

from DeepCache import DeepCacheSDHelper


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default = "sd1.5")
    parser.add_argument("--prompt", type=str, default='a photo of an astronaut on a moon')
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.model_type.lower() == 'sdxl':
        pipe = StableDiffusionXLPipeline.from_pretrained(
            'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to("cuda:0")
    elif args.model_type.lower() == 'svd':
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.enable_model_cpu_offload()
    elif args.model_type.lower() == 'sd1.5':
        pipe = StableDiffusionPipeline.from_pretrained(
            'runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16
        ).to("cuda:0")
    else:
        raise NotImplementedError

    prompt = args.prompt
    seed = args.seed

    if args.model_type.lower() == 'svd':
        from diffusers.utils import load_image, export_to_video
        image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")

        set_random_seed(42)
        frames = pipe(
            image, 
            decode_chunk_size=8,
        ).frames[0]
        export_to_video(frames, "{}_origin.mp4".format('rocket'), fps=7)

        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=3,
            cache_branch_id=1,
        )

        set_random_seed(42)
        frames = pipe(
            image, 
            decode_chunk_size=8,
        ).frames[0]
        export_to_video(frames, "{}_deepcache.mp4".format('rocket'), fps=7)
        helper.dismount()
    else:
        for _ in range(2):
            set_random_seed(seed)
            pipeline_output = pipe(
                prompt, 
                output_type='pt'
            ).images[0]

        helper = DeepCacheSDHelper(pipe=pipe)
        helper.set_params(
            cache_interval=3,
            cache_branch_id=10,
        )

        set_random_seed(seed)
        deepcache_pipeline_output = pipe(
                prompt,
                output_type='pt'
        ).images[0]

        save_image([pipeline_output, deepcache_pipeline_output], 'output.png')
        helper.dismount()
