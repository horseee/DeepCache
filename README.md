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
* 🚀 Support Stable Diffusion, Stable Diffusion XL, Stable Video Diffusion, Stable Diffusion Pipeline / XL Pipeline for Inpainting, Stable Diffusion Img2Img Pipeline, DDPM
* 🚀 Compatible with sampling algorithms like DDIM and PLMS

### Updates
* **January 5, 2024**: 💥A doc page for DeepCache has been added in [Diffusers](https://github.com/huggingface/diffusers)! Check [here](https://huggingface.co/docs/diffusers/main/en/optimization/deepcache) for more information. Many thanks to the Diffusers team!
* **December 26, 2023**: 🔥Update a plug-and-play implementation of DeepCache, no longer requiring any modifications of the diffuser's code! Check [here](https://github.com/horseee/DeepCache#usage) for the detailed usage! Big thanks to @[yuanshi9815](https://github.com/Yuanshi9815) for contributing the code!
* **December 25, 2023**: A demo is available via Colab [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1COM9tfGvHSJ8Zn4tScZCGgt8nxAEOlAz?usp=sharing)
* **December 21, 2023**: Release the code for **Stable Video Diffusion** and **Text2Video-Zero**. In the figure below, the upper row shows the original videos generated by SVD-XT, and the lower row is accelerated by DeepCache. For Text2Video-Zero, the results can be found [here](https://github.com/horseee/DeepCache/blob/master/assets/Text2Video-Zero.gif)
<div align="center">
  <img src="assets/svd.gif" width="90%" ></img>
  <br>
  <em>
      (1.7x acceleration of SVD-XT) 
  </em>
</div>


* **December 20, 2023**: Release the code for **DDPM**. See [here](https://github.com/horseee/DeepCache/tree/master/DeepCache#experiment-code-for-ddpm) for the experimental code and instructions.

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

### Install
```bash
pip install DeepCache
```

### Usage

```python
import torch

# Loading the original pipeline
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained('runwayml/stable-diffusion-v1-5', torch_dtype=torch.float16).to("cuda:0")

# Import the DeepCacheSDHelper
from DeepCache import DeepCacheSDHelper
helper = DeepCacheSDHelper(pipe=pipe)
helper.set_params(
    cache_interval=3,
    cache_branch_id=0,
)
helper.enable()

# Generate Image
deepcache_image = pipe(
        prompt,
        output_type='pt'
).images[0]
helper.disable()
```
We here take the Stable Diffusion pipeline as an example. You can replace pipe with any variants of the Stable Diffusion pipeline, including choices like SDXL, SVD, and more. You can find examples in the [script](https://github.com/horseee/DeepCache#a-general-script-for-sd). The argument `cache_branch_id` specifies the selected skip branch. For the skip branches that are deeper, the model will engage them only during the caching steps, and exclude them during the retrieval steps. The argument `cache_interval` represents the interval for updating the cache.

### A general script for SD
```bash
python main.py --model_type sdxl #Support [sdxl, sd1.5, sd2.1, svd, sd-inpaint, sdxl-inpaint, sd-img2img]
```

## Experimental code for DeepCache
The above implementation does not require changes to the `forward` or `__call__` functions in the Diffusers pipeline, and is, therefore, more general. The following section is the experimental code that can be used to reproduce the results in the paper. It is implemented one by one for different model structures and pipelines, and thus, may not work properly due to the update of diffusers.

### Setup
```
pip install diffusers==0.24.0 transformers
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

Currently, our code supports the models that can be loaded by [StableDiffusionPipeline](https://huggingface.co/docs/diffusers/v0.24.0/en/api/pipelines/stable_diffusion/text2img#diffusers.StableDiffusionPipeline). You can specify the model name by the argument `--model`, which by default, is `runwayml/stable-diffusion-v1-5`. 

### Stable Video Diffusion 
```bash
python stable_video_diffusion.py
```

<details>
  <summary>Output:</summary>
  
```bash
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00,  8.36it/s]
2023-12-21 04:56:47,329 - INFO - Running baseline...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [01:27<00:00,  3.49s/it]
2023-12-21 04:58:26,121 - INFO - Origin: 98.66 seconds
Loading pipeline components...: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 10.59it/s]
2023-12-21 04:58:27,202 - INFO - Running DeepCache...
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 25/25 [00:49<00:00,  1.96s/it]
2023-12-21 04:59:26,607 - INFO - DeepCache: 59.31 seconds
```

</details>

### DDPM and LDM
Please check [here](https://github.com/horseee/DeepCache/tree/master/experiments) for the experimental code of DDPM and LDM.


## Visualization

Images in the upper line are the baselines and the images in the lower line are accelerated by DeepCache. 

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

* Diffusers: https://huggingface.co/docs/diffusers/main/en/optimization/deepcache
* OneDiff Integration: https://github.com/siliconflow/onediff?tab=readme-ov-file#easy-to-use by @[SiliconFlow](https://github.com/siliconflow). OneDiff also has implementations for DeepCache on SVD, check [this](https://github.com/siliconflow/onediff/pull/438) for details.
* Comfyui: https://gist.github.com/laksjdjf/435c512bc19636e9c9af4ee7bea9eb86 by @[laksjdjf](https://github.com/laksjdjf)
* Colab & Gradio: https://github.com/camenduru/DeepCache-colab by @[camenduru](https://github.com/camenduru/DeepCache-colab)
* WebUI: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/14210 by @[aria1th](https://github.com/aria1th)
* SD.Next: https://github.com/vladmandic/automatic by @[vladmandic](https://github.com/vladmandic)
* MLX_deepcache for running SD on M1 Mac: https://github.com/Barry0121/mlx_deepcache by @[Barry Xue](https://github.com/Barry0121)

We warmly welcome contributions from everyone. Please feel free to reach out to us.


## Bibtex
```
@inproceedings{ma2023deepcache,
  title={DeepCache: Accelerating Diffusion Models for Free},
  author={Ma, Xinyin and Fang, Gongfan and Wang, Xinchao},
  booktitle={The IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```





