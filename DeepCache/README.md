
## Code

### TOC
- [Explanation of Code for SD/SDXL](#code-for-sd-and-sdxl)
- [Code for DDPM](#experiment-code-for-ddpm)

### Explanation of Code for SD and SDXL

We make some modifications to the code for SD and SDXL. Here we use the code of SD as an example to highlight the key changes:

1. `pipeline_stable_diffusion.py`:

Added `sample_from_quad_center` function (Lines 86-96) for non-uniform sampling, allowing selection of a specific number of steps from the total inference steps.
Modified the `__call__` of `StableDiffusionPipeline` (Lines 726-762) to incorporate feature caching in the outermost loop.

2. `pipeline_utils.py`:
Incorporated code (Lines 317-318) to load the model class specifically designed for DeepCache.

3. `unet_2d_block.py`:
Adjusted the forward functions of `CrossAttnDownBlock2D` and `CrossAttnUpBlock2D` to either reuse or cache features.

4. `unet_2d_condition.py`:
Altered the forward function (Lines 957-1142) to also reuse or cache features.

### Experiment Code for DDPM

#### Requirement
```
pip install accelerate lmdb scipy diffusers
```

#### Instructions
1. Sample Images:
```bash
cd DeepCache/ddpm

accelerate launch ddim.py --config configs/{DATASET_NAME}.yml --exp deepcache --fid --timesteps 100 --eta 0 --ni --use_pretrained --cache --cache_interval 5 --branch 2
```
Select `DATASET_NAME` from [<u>cifar10</u>, <u>bedroom</u>, <u>church</u>]. For the experiment on `CIFAR10`, add `--skip_type quad` to use the quadratic skip type. `cache_interval` here corresponds to `N` in `1:N` as specified in the paper, and `branch` represents the selected branch to perform the caching strategy. The value of `branch` starts from 0.  In our experiment, we set `branch=2` (the 3-th skip path in the model)

2. Testing FID:
```bash
python fid.py --path runtime_log/{YOUR_PATH_FOR_IMAGES}/images npz/cifar10_fid.npz
```
The pre-calculated npz archive for these three datasets can be downloaded [here](https://drive.google.com/file/d/1oAb3Jik40mExmUhWcF990IRDY5UvT1rh/view?usp=sharing). The npz archives are generated following [the instruction](https://github.com/mseitzer/pytorch-fid?tab=readme-ov-file#generating-a-compatible-npz-archive-from-a-dataset) in `pytorch-fid`.






