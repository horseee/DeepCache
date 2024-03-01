
# Code

## TOC
- [Experiment Code for DDPM](#experiment-code-for-ddpm)
- [Experiment Code for SD](#experiment-code-for-SD)
- [Experiment Code for LDM](#experiment-code-for-LDM)
- [MACs Calculation](#macs-calculation)

## Experiment Code for DDPM

#### Requirement
```
pip install accelerate lmdb scipy diffusers pytorch_fid
```

#### Instructions
1. Sample Images:
```bash
cd ddpm/
accelerate launch ddim.py --config configs/{DATASET_NAME}.yml --exp deepcache --fid --timesteps 100 --eta 0 --ni --use_pretrained --cache --cache_interval 5 --branch 2
```
Select `DATASET_NAME` from [<u>cifar10</u>, <u>bedroom</u>, <u>church</u>]. For the experiment on `CIFAR10`, add `--skip_type quad` to use the quadratic skip type. `cache_interval` here corresponds to `N` in `1:N` as specified in the paper, and `branch` represents the selected branch to perform the caching strategy. The value of `branch` starts from 0.  In our experiment, we set `branch=2` (the 3-th skip path in the model)

2. Testing FID:
```bash
python fid.py --path runtime_log/{YOUR_PATH_FOR_IMAGES}/images npz/cifar10_fid.npz
```
The pre-calculated npz archive for these three datasets can be downloaded [here](https://drive.google.com/file/d/1oAb3Jik40mExmUhWcF990IRDY5UvT1rh/view?usp=sharing). The npz archives are generated following [the instruction](https://github.com/mseitzer/pytorch-fid?tab=readme-ov-file#generating-a-compatible-npz-archive-from-a-dataset) in `pytorch-fid`.


## Experiment Code for SD

#### Requirement
```
pip install diffusers==0.24.0 transformers open_clip_torch
```

#### Instructions

* Step 1: Generate Images

For DeepCache:
```bash
python generate.py --dataset coco2017 --layer 0 --block 0 --update_interval 2 --uniform --steps 50 --batch_size 16 
```

For Baselines:
```bash
python generate.py --dataset coco2017  --original --steps 50 --batch_size 16 # For original pipeline
python generate.py --dataset coco2017 --bk base --steps 50 --batch_size 16 # For BK-SDM
```

* Step 2: Evaluate
```bash
python clip_score.py PATH_TO_SAVED_IMAGES
```

## Experiment Code for LDM
#### Requirement
Please follow [LDM](https://github.com/CompVis/latent-diffusion) to install the requirements. We have no extra requirements.

#### Instructions
Under re-organizing and testing. The code has been released, and we will soon update the instructions.



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








