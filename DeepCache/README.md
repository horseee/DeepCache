
# Code

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

