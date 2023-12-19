# DeepCache: Accelerating Diffusion Models for Free
<div align="center">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/example_compress.gif" width="100%" ></img>
  <br>
  <em>
      (Results on Stable Diffusion v1.5. Left: 50 PLMS steps. Right: 2.3x acceleration upon 50 PLMS steps) 
  </em>
</div>
<br>

> **DeepCache: Accelerating Diffusion Models for Free**   
> [Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)   
> [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore  
> 🥯[[Arxiv]](https://arxiv.org/abs/2312.00858)🎄[[Project Page]](https://horseee.github.io/Diffusion_DeepCache/)

### Why DeepCache  
* 🚀 Training-free and almost lossless
* 🚀 Support Stable Diffusion, Stable Diffusion XL
* 🚀 Compatible with sampling algorithms like DDIM and PLMS

### Updates
* **December 20, 2023**: Release the code for **DDPM**. See [here](https://github.com/horseee/DeepCache/tree/master/DeepCache#experiment-code-for-ddpm) for the code and instruction.
* **December 6, 2023**: Release the code for **Stable Diffusion XL**. The results of the `stabilityai/stable-diffusion-xl-base-1.0` are shown in the below figure, with the same prompts from the first figure.
<div align="center">
  <img src="assets/sdxl.png" width="90%" ></img>
  <br>
  <em>
      (2.6x acceleration of Stable Diffusion XL) 
  </em>
</div>


### Introduction
We introduce **DeepCache**, a novel **training-free and almost lossless** paradigm that accelerates diffusion models from the perspective of model architecture. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. DeepCache accelerates Stable Diffusion v1.5 by 2.3x with only a 0.05 decline in CLIP Score, and LDM-4-G(ImageNet) by 4.1x with a 0.22 decrease in FID.

<div align="center">
<img width="50%" alt="image" src="https://github.com/horseee/DeepCache/assets/18592211/9ce3930c-c84c-4af8-8c6a-b6803a5a7b1d">
</div>

## Quick Start

### Requirements
```bash
pip install transformers diffusers
```

### Stable Diffusion XL
```bash
python stable_diffusion_xl.py --model stabilityai/stable-diffusion-xl-base-1.0
```


<details>
<summary>Output:</summary>

```bash
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  6.62it/s]
2023-12-06 01:44:28,578 - INFO - Running baseline...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:17<00:00,  2.93it/s]
2023-12-06 01:44:46,095 - INFO - Baseline: 17.52 seconds
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7/7 [00:00<00:00,  8.06it/s]
2023-12-06 01:45:02,865 - INFO - Running DeepCache...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:06<00:00,  8.01it/s]
2023-12-06 01:45:09,573 - INFO - DeepCache: 6.71 seconds
2023-12-06 01:45:10,678 - INFO - Saved to output.png. Done!
```

</details>

You can add `--refine` at the end of the command to activate the refiner model for SDXL.

### Stable Diffusion v1.5
```bash
python stable_diffusion.py --model runwayml/stable-diffusion-v1-5
```

<details>
<summary>Output:</summary>

```bash
2023-12-03 16:18:13,636 - INFO - Loaded safety_checker as StableDiffusionSafetyChecker from `safety_checker` subfolder of runwayml/stable-diffusion-v1-5.
2023-12-03 16:18:13,699 - INFO - Loaded vae as AutoencoderKL from `vae` subfolder of runwayml/stable-diffusion-v1-5.
Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.88it/s]
2023-12-03 16:18:22,837 - INFO - Running baseline...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 15.33it/s]
2023-12-03 16:18:26,174 - INFO - Baseline: 3.34 seconds
2023-12-03 16:18:26,174 - INFO - Running DeepCache...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 34.06it/s]
2023-12-03 16:18:27,718 - INFO - DeepCache: 1.54 seconds
2023-12-03 16:18:27,935 - INFO - Saved to output.png. Done!
```

</details>

### Stable Diffusion v2.1

```bash
python stable_diffusion.py --model stabilityai/stable-diffusion-2-1
```

<details>
  <summary>Output:</summary>
  
```bash
2023-12-03 16:21:17,858 - INFO - Loaded feature_extractor as CLIPImageProcessor from `feature_extractor` subfolder of stabilityai/stable-diffusion-2-1.
2023-12-03 16:21:17,864 - INFO - Loaded scheduler as DDIMScheduler from `scheduler` subfolder of stabilityai/stable-diffusion-2-1.
Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  5.35it/s]
2023-12-03 16:21:49,770 - INFO - Running baseline...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:14<00:00,  3.42it/s]
2023-12-03 16:22:04,551 - INFO - Baseline: 14.78 seconds
2023-12-03 16:22:04,551 - INFO - Running DeepCache...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:08<00:00,  6.10it/s]
2023-12-03 16:22:12,911 - INFO - DeepCache: 8.36 seconds
2023-12-03 16:22:13,417 - INFO - Saved to output.png. Done!
```

</details>

Currently, our code supports the models that can be loaded by [StableDiffusionPipeline](https://huggingface.co/docs/diffusers/v0.24.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline). You can specify the model name by the argument `--model`, which by default, is `runwayml/stable-diffusion-v1-5`. We are arranging the code for LDM and DDPM and will release it in the next few days.

### Usage

* For Stable Diffusion XL
```python
import torch
from DeepCache import StableDiffusionXLPipeline as DeepCacheStableDiffusionXLPipeline
pipe = DeepCacheStableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda:0")
prompt = "A photo of a cat. Focus light and create sharp, defined edges."        
deepcache_output = pipe(
    prompt, 
    cache_interval=3, cache_layer_id=0, cache_block_id=0,
    output_type='pt', return_dict=True
).images
```

* For Stable Diffusion 
```python
import torch
from DeepCache import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16).to("cuda:0")
prompt = "a photo of an astronaut on a moon"
deepcache_output = pipe(
    prompt, 
    cache_interval=5, cache_layer_id=0, cache_block_id=0,
    uniform=True, #pow=1.4, center=15, # only for uniform = False
    output_type='pt', return_dict=True
).images
```

Arguments:
* **cache_interval**: the interval (N in the 1:N strategy) of cache update. Larger intervals bring more significant speedup.
* **cache_layer_id & cache_block_id**: the block/layer ID of the selected skip branch. 
* **uniform**: whether to enable the uniform caching strategy.
* **pow & center**: the hyperparameters for non-uniform 1:N strategy.


## Visualization

Images in the upper line are the baselines, and the images in the lower line are accelerated by DeepCache. 

### Stable Diffusion v1.5 (2.15x Acceleration)
<div align="center">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/sd_examples_high_res.png" width="100%">
</div>

### LDM-4-G for ImageNet (6.96x Acceleration)
<div align="center">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/imagenet.png" width="100%">
</div>

### DDPM for LSUN Church & Bedroom (1.48x Acceleration)
<div align="center">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/bedroom.png" width="100%">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/church.png" width="100%">
</div>

## Quantitative Results

### ImageNet
<div align="center">
<img width="80%" alt="image" src="https://github.com/horseee/DeepCache/assets/18592211/151d7639-2501-45cf-8de5-2af3bb5a354b">
</div>


### Stable Diffusion v1.5

<div align="center">
<img width="80%" alt="image" src="https://github.com/horseee/DeepCache/assets/18592211/e9bd7a8e-07c8-4296-95a2-12d008995807">
</div>

More results can be found in our [paper](https://arxiv.org/abs/2312.00858)

## Other Implementations and Plugins
We sincerely thank the authors listed below who implemented DeepCache in plugins or other contexts. 

* OneDiff Integration: https://github.com/Oneflow-Inc/onediff/pull/426 by @[Oneflow-Inc](https://github.com/Oneflow-Inc)
* Comfyui: https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86 by @[laksjdjf](https://github.com/laksjdjf)
* Colab & Gradio: https://github.com/camenduru/DeepCache-colab by @[camenduru](https://github.com/camenduru/DeepCache-colab)
* WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14210 by @[aria1th](https://github.com/aria1th)

We warmly welcome contributions from everyone. Please feel free to reach out to us.


## Bibtex
```
@article{ma2023deepcache,
  title={DeepCache: Accelerating Diffusion Models for Free},
  author={Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  journal={arXiv preprint arXiv:2312.00858},
  year={2023}
}
```





