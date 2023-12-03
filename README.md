# DeepCache: Accelerating Diffusion Models for Free
<div align="center">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/example.gif" width="100%"></img>
</div>

> **DeepCache: Accelerating Diffusion Models for Free**   
> [Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)   
> [Learning and Vision Lab](http://lv-nus.org/), National University of Singapore  
> [[Arxiv]]() [[Project Page]](https://horseee.github.io/Diffusion_DeepCache/)

### Introduction

Diffusion models have recently gained unprecedented attention in the field of image synthesis due to their remarkable generative capabilities. Notwithstanding their prowess, these models often incur substantial computational costs, primarily attributed to the sequential denoising process and cumbersome model size. Traditional methods for compressing diffusion models typically involve extensive retraining, presenting cost and feasibility challenges. In this paper, we introduce DeepCache, a novel training-free paradigm that accelerates diffusion models from the perspective of model architecture. DeepCache capitalizes on the inherent temporal redundancy observed in the sequential denoising steps of diffusion models, which caches and retrieves features across adjacent denoising stages, thereby curtailing redundant computations. Utilizing the property of the U-Net, we reuse the high-level features while updating the low-level features in a very cheap way. This innovative strategy, in turn, enables a speedup factor of 2.3X for Stable Diffusion v1.5 with only a 0.05 decline in CLIP Score, and 4.1X for LDM-4-G with a slight decrease of 0.22 in FID on ImageNet. Our experiments also demonstrate DeepCache's superiority over existing pruning and distillation methods that necessitate retraining and its compatibility with current sampling techniques. Furthermore, we find that under the same throughput, DeepCache effectively achieves comparable or even marginally improved results with DDIM or PLMS.


### Core Idea: Feature Reusing

The high-level features of U-Net can be heavily reused across the denoising process. In this case, only the shallow layers will be executed. 
<div align="center">
<img width="50%" alt="image" src="https://github.com/horseee/DeepCache/assets/18592211/9ce3930c-c84c-4af8-8c6a-b6803a5a7b1d">
</div>

## Quick Start

```bash
python stable_diffusion.py
```

Output:
```bash

```




## Visualization

The original images are depicted in the upper line, and the images accelerated by DeepCache are shown in the lower line

### Stable Diffusion v1.5 (2.15x)
<div align="center">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/sd_examples_high_res.png" width="100%">
</div>

### LDM-4-G for ImageNet (6.96x)
<div align="center">
  <img src="https://github.com/horseee/Diffusion_DeepCache/blob/master/static/images/imagenet.png" width="100%">
</div>

### DDPM for LSUN Church & Bedroom (1.48x)
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





