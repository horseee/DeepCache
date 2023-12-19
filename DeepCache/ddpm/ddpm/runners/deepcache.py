import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data
from torch.nn.functional import adaptive_avg_pool2d

from ..models.ema import EMAHelper
from ..functions import get_optimizer
from ..functions.losses import loss_registry
from ..datasets import get_dataset, data_transform, inverse_data_transform
from ..functions.ckpt_util import get_ckpt_path

import torchvision.utils as tvu

from ..utils import tools


def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


class Diffusion(object):
    def __init__(self, args, config):
        self.args = args
        self.config = config
        self.logger = args.logger
        self.accelerator = args.accelerator
        self.device = self.accelerator.device
        self.config.device = self.device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), alphas_cumprod[:-1]], dim=0
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
            # torch.cat(
            # [posterior_variance[1:2], betas[1:]], dim=0).log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()


    def sample(self):
        if self.args.cache:
            from ..models.deepcache_diffusion import Model
            model = Model(self.config)
            self.logger.log('Sampling in DeepCache mode')
        else:
            from ..models.diffusion import Model
            model = Model(self.config)
       
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.logger.checkpoint_path, "ckpt.pth"),
                    map_location='cpu',
                )
                self.logger.log("Loading from latest checkpoint: {}".format(
                    os.path.join(self.logger.checkpoint_path, "ckpt.pth")
                ))
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location='cpu',
                )
                self.logger.log("Loading from latest checkpoint: {}".format(
                    os.path.join(self.logger.checkpoint_path, f"ckpt_{self.config.sampling.ckpt_id}.pth")
                ))
            model.load_state_dict(tools.unwrap_module(states[0]), strict=True)
            
            if self.config.model.ema:
                ema_helper = EMAHelper(mu=self.config.model.ema_rate)
                ema_helper.register(model)
                ema_helper.load_state_dict(tools.unwrap_module(states[-1]))
                ema_helper.ema(model)
            else:
                ema_helper = None
            
            model = self.accelerator.prepare(model)
        else:
            if self.config.data.dataset == "CIFAR10":
                name = "cifar10"
            elif self.config.data.dataset == "LSUN":
                name = f"lsun_{self.config.data.category}"
            else:
                raise ValueError
            ckpt = get_ckpt_path(f"ema_{name}")
            self.logger.log("Loading checkpoint {}".format(ckpt))
            msg = model.load_state_dict(torch.load(ckpt, map_location=self.device), strict=False)

            self.logger.log(msg)
            model = self.accelerator.prepare(model)

        model.eval()

        if self.args.fid:
            self.sample_fid(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, total_n_samples=50000, save_images = True, timesteps=None):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        self.logger.log(f"starting from image {img_id}")
        total_n_samples = total_n_samples // self.accelerator.num_processes
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        generate_samples = []
        throughput = []
        sample_start_time = time.time()
        with torch.no_grad(), tqdm.tqdm(range(n_rounds)) as t:
            for _ in t:
                start_time = time.time()
                n = config.sampling.batch_size
                x = torch.randn(
                    n,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model, timesteps=timesteps)
                x = inverse_data_transform(config, x)

                use_time = time.time() - start_time
                throughput.append(x.shape[0] / use_time)
                t.set_description(f"Throughput: {np.mean(throughput):.2f} samples/s")
                
                if save_images:
                    for i in range(n):
                        tvu.save_image(
                            x[i], os.path.join(self.args.image_folder, f"{self.accelerator.process_index}_{img_id}.png")
                        )
                        img_id += 1
                else:
                    generate_samples.append(x)
        
        self.args.accelerator.wait_for_everyone()
        self.logger.log(f"Time taken: {time.time() - sample_start_time} seconds")
        return generate_samples

    
    def sample_image(self, x, model, last=True, timesteps=None):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        if timesteps is None:
            timesteps = self.args.timesteps
        #print(self.args.sample_type, self.args.skip_type, timesteps)

        if self.args.sample_type == "generalized":
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            
            from ..functions.deepcache_denoising import generalized_steps
            xs = generalized_steps(
                x, seq, model, self.betas, 
                timesteps=timesteps,
                cache_interval=self.args.cache_interval,  # for uniform
                non_uniform=self.args.non_uniform, pow = self.args.pow, center = self.args.center, branch=self.args.branch,  # for non-uniform
                eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
            # Not implemented for DeepCache
            if self.args.skip_type == "uniform":
                skip = self.num_timesteps // timesteps
                seq = range(0, self.num_timesteps, skip)
            elif self.args.skip_type == "quad":
                seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), timesteps
                    )
                    ** 2
                )
                seq = [int(s) for s in list(seq)]
            else:
                raise NotImplementedError
            from ..functions.deepcache_denoising import ddpm_steps
            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x





        
