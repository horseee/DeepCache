import os
import logging
import time
import glob

import numpy as np
import tqdm
import torch
import torch.utils.data as data

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

    def train(self):
        args, config = self.args, self.config
        #tb_logger = self.config.tb_logger
        dataset, test_dataset = get_dataset(args, config)
        train_loader = data.DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        if self.args.dyn:
            from ..models.dyndiffusion import Model
            model = Model(config)
        else:
            from ..models.diffusion import Model
            model = Model(config)

        optimizer = get_optimizer(self.config, model.parameters())    
        start_epoch, step = 0, 0
        if self.args.resume_training:

            if args.use_pretrained:
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
            else:
                states = torch.load(os.path.join(self.args.log_path, "ckpt.pth"))
                model.load_state_dict(states[0])

                states[1]["param_groups"][0]["eps"] = self.config.optim.eps
                optimizer.load_state_dict(states[1])
                start_epoch = states[2]
                step = states[3]
                if self.config.model.ema:
                    ema_helper.load_state_dict(states[4])

        if torch.cuda.device_count() > 1:
            model, optimizer, train_loader = self.accelerator.prepare(
                model, optimizer, train_loader
            )
            device = self.accelerator.device
        else:
            model = model.to(self.device)
            model = torch.nn.DataParallel(model)
            device = self.device
        
        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(model)
        else:
            ema_helper = None
        
        #model.eval()
        #rng_status = tools.save_and_set_random_status(seed=1234+self.accelerator.process_index)
        #self.test(model, total_n_samples=50000, timesteps=10, save_id=str('start'))
        #tools.restore_random_status(rng_status)

        for epoch in tqdm.tqdm(range(start_epoch, self.config.training.n_epochs)):
            data_start = time.time()
            data_time = 0
          
            for i, (x, y) in enumerate(train_loader):
                optimizer.zero_grad()

                n = x.size(0)
                data_time += time.time() - data_start
                model.train()
                step += 1

                x = x.to(device)
                x = data_transform(self.config, x)
                e = torch.randn_like(x)
                b = self.betas

                # antithetic sampling
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = loss_registry[config.model.type](model, x, t, e, b)

                #tb_logger.add_scalar("loss", loss, global_step=step)

                self.logger.log(
                    f"step: {step}, loss: {loss.item()}, data time: {data_time / (i+1)}"
                )
                
                if torch.cuda.device_count() > 1:
                    self.accelerator.backward(loss)
                else:
                    loss.backward()

                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.optim.grad_clip
                    )
                except Exception:
                    pass
                optimizer.step()

                if self.config.model.ema:
                    ema_helper.update(model)
                    
                if self.accelerator.is_main_process and (step % self.config.training.snapshot_freq == 0 or step == 1) :
                    states = [
                        model.state_dict(),
                        optimizer.state_dict(),
                        epoch,
                        step,
                    ]
                    if self.config.model.ema:
                        states.append(ema_helper.state_dict())

                    torch.save(
                        states,
                        os.path.join(self.logger.checkpoint_path, "ckpt_{}.pth".format(step)),
                    )
                    torch.save(states, os.path.join(self.logger.checkpoint_path, "ckpt.pth"))
                
                data_start = time.time()

            if epoch % 10 == 0: 
                model.eval()
                ema_copy = ema_helper.ema_copy(model, self.accelerator)
                ema_copy.eval()
                rng_status = tools.save_and_set_random_status(seed=1234+self.accelerator.process_index)
                self.test(ema_copy, total_n_samples=50000, timesteps=10, save_id=str(epoch))
                tools.restore_random_status(rng_status)
        
                del ema_copy
                torch.cuda.empty_cache()

            

    def sample(self):
        from ..models.diffusion import Model
        model = Model(self.config)
        
        if not self.args.use_pretrained:
            if getattr(self.config.sampling, "ckpt_id", None) is None:
                states = torch.load(
                    os.path.join(self.logger.checkpoint_path, "ckpt.pth"),
                    map_location='cpu',
                )
            else:
                states = torch.load(
                    os.path.join(
                        self.args.log_path, f"ckpt_{self.config.sampling.ckpt_id}.pth"
                    ),
                    map_location='cpu',
                )

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
            # This used the pretrained DDPM model, see https://github.com/pesser/pytorch_diffusion
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
        elif self.args.interpolation:
            self.sample_interpolation(model)
        elif self.args.sequence:
            self.sample_sequence(model)
        else:
            raise NotImplementedError("Sample procedeure not defined")

    def sample_fid(self, model, total_n_samples=50000, save_images = True):
        config = self.config
        img_id = len(glob.glob(f"{self.args.image_folder}/*"))
        self.logger.log(f"starting from image {img_id}")
        total_n_samples = total_n_samples // self.accelerator.num_processes
        n_rounds = (total_n_samples - img_id) // config.sampling.batch_size

        throughput = []
        generate_samples = []
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
                x = self.sample_image(x, model)
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
        return generate_samples

    def sample_sequence(self, model):
        config = self.config

        x = torch.randn(
            8,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )

        # NOTE: This means that we are producing each predicted x0, not x_{t-1} at timestep t.
        with torch.no_grad():
            _, x = self.sample_image(x, model, last=False)

        x = [inverse_data_transform(config, y) for y in x]

        for i in range(len(x)):
            for j in range(x[i].size(0)):
                tvu.save_image(
                    x[i][j], os.path.join(self.args.image_folder, f"{j}_{i}.png")
                )

    def sample_interpolation(self, model):
        config = self.config

        def slerp(z1, z2, alpha):
            theta = torch.acos(torch.sum(z1 * z2) / (torch.norm(z1) * torch.norm(z2)))
            return (
                torch.sin((1 - alpha) * theta) / torch.sin(theta) * z1
                + torch.sin(alpha * theta) / torch.sin(theta) * z2
            )

        z1 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        z2 = torch.randn(
            1,
            config.data.channels,
            config.data.image_size,
            config.data.image_size,
            device=self.device,
        )
        alpha = torch.arange(0.0, 1.01, 0.1).to(z1.device)
        z_ = []
        for i in range(alpha.size(0)):
            z_.append(slerp(z1, z2, alpha[i]))

        x = torch.cat(z_, dim=0)
        xs = []

        # Hard coded here, modify to your preferences
        with torch.no_grad():
            for i in range(0, x.size(0), 8):
                xs.append(self.sample_image(x[i : i + 8], model))
        x = inverse_data_transform(config, torch.cat(xs, dim=0))
        for i in range(x.size(0)):
            tvu.save_image(x[i], os.path.join(self.args.image_folder, f"{i}.png"))

    def sample_image(self, x, model, last=True, timesteps=None):
        try:
            skip = self.args.skip
        except Exception:
            skip = 1
        
        if timesteps is None:
            timesteps = self.args.timesteps

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
            from ..functions.denoising import generalized_steps

            xs = generalized_steps(x, seq, model, self.betas, eta=self.args.eta)
            x = xs
        elif self.args.sample_type == "ddpm_noisy":
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
            from ..functions.denoising import ddpm_steps

            x = ddpm_steps(x, seq, model, self.betas)
        else:
            raise NotImplementedError
        if last:
            x = x[0][-1]
        return x

    def test(self, model, total_n_samples, timesteps, save_id = None):
        config = self.config
        batch_size = 250
       
        total_n_samples = total_n_samples // self.accelerator.num_processes
        n_rounds = total_n_samples // batch_size

        generate_samples = []

        with torch.no_grad():
            for _ in tqdm.tqdm(
                range(n_rounds), desc="Generating image samples for FID evaluation."
            ):
                n = config.sampling.batch_size
                x = torch.randn(
                    batch_size,
                    config.data.channels,
                    config.data.image_size,
                    config.data.image_size,
                    device=self.device,
                )

                x = self.sample_image(x, model, timesteps=timesteps, )
                generate_samples.append(x)

        generate_samples = torch.stack(generate_samples, 0).to(self.device)
        all_samples = self.accelerator.gather(generate_samples)
        all_samples = all_samples.view(-1, *all_samples.size()[-3:])
        if self.accelerator.is_main_process:
            all_samples = inverse_data_transform(config, all_samples)
            img_id = 0
            for i in range(all_samples.shape[0]):
                tvu.save_image(
                    all_samples[i], os.path.join(self.args.image_folder, f"{self.accelerator.process_index}_{img_id}.png")
                )
                img_id += 1
        
            fid = calculate_fid(
                [self.args.ref_npz, all_samples],  
                batch_size=50, device=self.device, dims=2048
            )
            self.logger.log("FID = {}".format(fid))

            if save_id is not None:
                tvu.save_image(
                    all_samples[:100], os.path.join(self.logger.image_ckpt_path, f"sample_{save_id}.png")
                )
            print("generate image in original code")
        self.accelerator.wait_for_everyone()   
