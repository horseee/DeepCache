class DeepCacheSDHelper(object):
    def __init__(self, pipe=None):
        if pipe is not None: self.pipe = pipe

    def enable(self, pipe=None):
        assert self.pipe is not None
        self.reset_states()
        self.wrap_modules()

    def disable(self):
        self.unwrap_modules()
        self.reset_states()
    
    def set_params(self,cache_interval=1, cache_branch_id=0, skip_mode='uniform'):
        cache_layer_id = cache_branch_id % 3
        cache_block_id = cache_branch_id // 3
        self.params = {
            'cache_interval': cache_interval,
            'cache_layer_id': cache_layer_id,
            'cache_block_id': cache_block_id,
            'skip_mode': skip_mode
        }

    def is_skip_step(self, block_i, layer_i, blocktype = "down"):
        self.start_timestep = self.cur_timestep if self.start_timestep is None else self.start_timestep # For some pipeline that the first timestep != 0
        cache_interval, cache_layer_id, cache_block_id, skip_mode = \
            self.params['cache_interval'], self.params['cache_layer_id'], self.params['cache_block_id'], self.params['skip_mode']
        if skip_mode == 'uniform':
            if (self.cur_timestep-self.start_timestep) % cache_interval == 0: return False
        if block_i > cache_block_id or blocktype == 'mid':
            return True
        if block_i < cache_block_id: return False
        return layer_i >= cache_layer_id if blocktype == 'down' else layer_i > cache_layer_id
        
    def is_enter_position(self, block_i, layer_i):
        return block_i == self.params['cache_block_id'] and layer_i == self.params['cache_layer_id']

    def wrap_unet_forward(self):
        self.function_dict['unet_forward'] = self.pipe.unet.forward
        def wrapped_forward(*args, **kwargs):
            self.cur_timestep = list(self.pipe.scheduler.timesteps).index(args[1].item())
            result = self.function_dict['unet_forward'](*args, **kwargs)
            return result
        self.pipe.unet.forward = wrapped_forward

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype = "down"):
        self.function_dict[
            (blocktype, block_name, block_i, layer_i)
        ] = block.forward
        def wrapped_forward(*args, **kwargs):
            skip = self.is_skip_step(block_i, layer_i, blocktype)
            result = self.cached_output[(blocktype, block_name, block_i, layer_i)] if skip else self.function_dict[(blocktype, block_name,  block_i, layer_i)](*args, **kwargs)
            if not skip: self.cached_output[(blocktype, block_name, block_i, layer_i)] = result
            return result
        block.forward = wrapped_forward

    def wrap_modules(self):
        # 1. wrap unet forward
        self.wrap_unet_forward()
        # 2. wrap downblock forward
        for block_i, block in enumerate(self.pipe.unet.down_blocks):
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                self.wrap_block_forward(attention, "attentions", block_i, layer_i)
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                self.wrap_block_forward(resnet, "resnet", block_i, layer_i)
            for downsampler in getattr(block, "downsamplers", []) if block.downsamplers else []:
                self.wrap_block_forward(downsampler, "downsampler", block_i, len(getattr(block, "resnets", [])))
            self.wrap_block_forward(block, "block", block_i, 0, blocktype = "down")
        # 3. wrap midblock forward
        self.wrap_block_forward(self.pipe.unet.mid_block, "mid_block", 0, 0, blocktype = "mid")
        # 4. wrap upblock forward
        block_num = len(self.pipe.unet.up_blocks)
        for block_i, block in enumerate(self.pipe.unet.up_blocks):
            layer_num = len(getattr(block, "resnets", []))
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                self.wrap_block_forward(attention, "attentions", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                self.wrap_block_forward(resnet, "resnet", block_num - block_i - 1, layer_num - layer_i - 1, blocktype = "up")
            for upsampler in getattr(block, "upsamplers", []) if block.upsamplers else []:
                self.wrap_block_forward(upsampler, "upsampler", block_num - block_i - 1, 0, blocktype = "up")
            self.wrap_block_forward(block, "block", block_num - block_i - 1, 0, blocktype = "up")

    def unwrap_modules(self):
        # 1. unet forward
        self.pipe.unet.forward = self.function_dict['unet_forward']
        # 2. downblock forward
        for block_i, block in enumerate(self.pipe.unet.down_blocks):
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                attention.forward = self.function_dict[("down", "attentions", block_i, layer_i)]
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                resnet.forward = self.function_dict[("down", "resnet", block_i, layer_i)]
            for downsampler in getattr(block, "downsamplers", []) if block.downsamplers else []:
                downsampler.forward = self.function_dict[("down", "downsampler", block_i, len(getattr(block, "resnets", [])))]
            block.forward = self.function_dict[("down", "block", block_i, 0)]
        # 3. midblock forward
        self.pipe.unet.mid_block.forward = self.function_dict[("mid", "mid_block", 0, 0)]
        # 4. upblock forward
        block_num = len(self.pipe.unet.up_blocks)
        for block_i, block in enumerate(self.pipe.unet.up_blocks):
            layer_num = len(getattr(block, "resnets", []))
            for (layer_i, attention) in enumerate(getattr(block, "attentions", [])):
                attention.forward = self.function_dict[("up", "attentions", block_num - block_i - 1, layer_num - layer_i - 1)]
            for (layer_i, resnet) in enumerate(getattr(block, "resnets", [])):
                resnet.forward = self.function_dict[("up", "resnet", block_num - block_i - 1, layer_num - layer_i - 1)]
            for upsampler in getattr(block, "upsamplers", []) if block.upsamplers else []:
                upsampler.forward = self.function_dict[("up", "upsampler", block_num - block_i - 1, 0)]
            block.forward = self.function_dict[("up", "block", block_num - block_i - 1, 0)]

    def reset_states(self):
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None
