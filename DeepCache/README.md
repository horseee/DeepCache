
# Code

## TOC
- [Explanation of Code for SD/SDXL](#code-for-sd-and-sdxl)
- [Code for DDPM](#experiment-code-for-ddpm)
- [MACs Calculation](#macs-calculation)

## Explanation of Code for SD and SDXL

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

## Experiment Code for DDPM

#### Requirement
```
pip install accelerate lmdb scipy diffusers pytorch_fid
```

#### Instructions
1. Sample Images:
```bash
cd DeepCache/ddpm/
accelerate launch ddim.py --config configs/{DATASET_NAME}.yml --exp deepcache --fid --timesteps 100 --eta 0 --ni --use_pretrained --cache --cache_interval 5 --branch 2
```
Select `DATASET_NAME` from [<u>cifar10</u>, <u>bedroom</u>, <u>church</u>]. For the experiment on `CIFAR10`, add `--skip_type quad` to use the quadratic skip type. `cache_interval` here corresponds to `N` in `1:N` as specified in the paper, and `branch` represents the selected branch to perform the caching strategy. The value of `branch` starts from 0.  In our experiment, we set `branch=2` (the 3-th skip path in the model)

2. Testing FID:
```bash
python fid.py --path runtime_log/{YOUR_PATH_FOR_IMAGES}/images npz/cifar10_fid.npz
```
The pre-calculated npz archive for these three datasets can be downloaded [here](https://drive.google.com/file/d/1oAb3Jik40mExmUhWcF990IRDY5UvT1rh/view?usp=sharing). The npz archives are generated following [the instruction](https://github.com/mseitzer/pytorch-fid?tab=readme-ov-file#generating-a-compatible-npz-archive-from-a-dataset) in `pytorch-fid`.

## MACs Calculation
If you want to calculate the FLOPs for each model, and also the FLOPs of the partial model executed in DeepCache, you can use the following code snippet, insert it before the iteration of denoising, and get the MACs of the model. Here are two examples:

* For DDPM: (insert it in [Line 153](https://github.com/horseee/DeepCache/blob/fb0ec94e046068eceebe185b2f5cada55b11be1e/DeepCache/ddpm/ddpm/runners/deepcache.py#L153) for DDPM pipeline)
```python
import sys
sys.path.append('../')
from flops import count_ops_and_params
example_inputs = {
    'x': torch.randn(1, 3, self.config.data.image_size, self.config.data.image_size).to(self.device), 
    't': torch.ones(1).to(self.device),
    'prv_f': [torch.randn(1, 256, 16, 16).to(self.device)],
    'branch': 2
}
macs, nparams = count_ops_and_params(model, example_inputs=example_inputs, layer_wise=False)
self.logger.log("#Params: {:.4f} M".format(nparams/1e6))
self.logger.log("#MACs: {:.4f} G".format(macs/1e9))
exit()
```
You can enable `layer_wise` to `True` to display the FLOPs for each module. If you are using a different branch for caching, you might need to check and update the `prv_f` shape.

* For Stable Diffusion: (insert it in [Line 752](https://github.com/horseee/DeepCache/blob/fb0ec94e046068eceebe185b2f5cada55b11be1e/DeepCache/sd/pipeline_stable_diffusion.py#L752) for SD pipeline)
```python
from ..flops import count_ops_and_params
example_inputs = {
    'sample': latent_model_input, 
    'timestep': t,
    'encoder_hidden_states': prompt_embeds,
    'cross_attention_kwargs': cross_attention_kwargs,
    'replicate_prv_feature': prv_features,
    'quick_replicate': cache_interval>1,
    'cache_layer_id': cache_layer_id,
    'cache_block_id': cache_block_id,
    'return_dict': False,
}
macs, nparams = count_ops_and_params(self.unet, example_inputs=example_inputs, layer_wise=False)
print("#Params: {:.4f} M".format(nparams/1e6))
print("#MACs: {:.4f} G".format(macs/1e9))
exit() 
```
To view the FLOPs for each module, you can set the `layer_wise` parameter to `True`. Additionally, if you want to see the FLOPs for the partial model executed during the retrieve steps, you can find the results in the second step with `cache_interval` larger than 2.








