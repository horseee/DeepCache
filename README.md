# DeepCache: Accelerating Diffusion Models for Free
<div align="center">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/example.gif" width="100%" ></img>
</div>


> **DeepCache: Accelerating Diffusion Models for Free**   
> [Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)   
> [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore  
> [[Arxiv]]() [[Project Page]](https://horseee.github.io/Diffusion_DeepCache/)

### Introduction

Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3X for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1X for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS.


### Core Idea: Feature Reusing

The high-level features in U-Net can be heavily reused across the denoising process. This allows us to only forward the shallow layers for most timesteps.
<div align="center">
<img width="70%" alt="image" src="https://github.com/horseee/DeepCache/assets/18592211/9ce3930c-c84c-4af8-8c6a-b6803a5a7b1d">
</div>

## Quick Start

### Stable Diffusion v1.5
```bash
python stable_diffusion.py --model runwayml/stable-diffusion-v1-5
```

Output:
```bash
2023-12-03 16:18:13,636 - INFO - Loaded safety_checker as StableDiffusionSafetyChecker from `safety_checker` subfolder of runwayml/stable-diffusion-v1-5.
Loading pipeline components...:  86%|████████████████████████████████████████████████████████▌         | 6/7 [00:01<00:00,  4.96it/s]2023-12-03 16:18:13,699 - INFO - Loaded vae as AutoencoderKL from `vae` subfolder of runwayml/stable-diffusion-v1-5.
Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████| 7/7 [00:01<00:00,  5.88it/s]
2023-12-03 16:18:14,858 - INFO - Warming up GPU...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:04<00:00, 11.91it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 15.36it/s]
2023-12-03 16:18:22,837 - INFO - Running baseline...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:03<00:00, 15.33it/s]
2023-12-03 16:18:26,174 - INFO - Baseline: 3.34 seconds
2023-12-03 16:18:26,174 - INFO - Running DeepCache...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:01<00:00, 34.06it/s]
2023-12-03 16:18:27,718 - INFO - DeepCache: 1.54 seconds
2023-12-03 16:18:27,935 - INFO - Saved to output.png. Done!
```

### Stable Diffusion v2.1

```bash
python stable_diffusion.py --model stabilityai/stable-diffusion-2-1
```

Output:
```
2023-12-03 16:21:17,858 - INFO - Loaded feature_extractor as CLIPImageProcessor from `feature_extractor` subfolder of stabilityai/stable-diffusion-2-1.
2023-12-03 16:21:17,864 - INFO - Loaded scheduler as DDIMScheduler from `scheduler` subfolder of stabilityai/stable-diffusion-2-1.
Loading pipeline components...: 100%|██████████████████████████████████████████████████████████████████| 6/6 [00:01<00:00,  5.35it/s]
2023-12-03 16:21:19,057 - INFO - Warming up GPU...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:15<00:00,  3.23it/s]
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:14<00:00,  3.43it/s]
2023-12-03 16:21:49,770 - INFO - Running baseline...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:14<00:00,  3.42it/s]
2023-12-03 16:22:04,551 - INFO - Baseline: 14.78 seconds
2023-12-03 16:22:04,551 - INFO - Running DeepCache...
100%|████████████████████████████████████████████████████████████████████████████████████████████████| 50/50 [00:08<00:00,  6.10it/s]
2023-12-03 16:22:12,911 - INFO - DeepCache: 8.36 seconds
2023-12-03 16:22:13,417 - INFO - Saved to output.png. Done!
```


## Visualization

The original images are depicted in the upper line, and the images accelerated by DeepCache are shown in the lower line

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

## Bibtex
```
```





