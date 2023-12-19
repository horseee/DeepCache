import argparse
import traceback
import shutil
import logging
import yaml
import random
import sys
import os
import torch
import numpy as np

from ddpm.utils.logging import Logger, EmptyLogger
from ddpm.utils.tools import set_random_seed
from accelerate import Accelerator, DistributedDataParallelKwargs

torch.set_printoptions(sci_mode=False)

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    
    parser.add_argument(
        "--test", action="store_true", help="Whether to test the model"
    )
    parser.add_argument(
        "--sample", action="store_true", help="Whether to produce samples from the model",
    )
    parser.add_argument(
        "--image_folder", type=str, default="images", help="folder name for storing the sampled images"
    )

    parser.add_argument(
        "--fid", action="store_true"
    )
    parser.add_argument(
        "--interpolation", action="store_true"
    )
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "--ni", action="store_true", help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--use_pretrained", action="store_true"
    )
    parser.add_argument(
        "--sample_type", type=str, default="generalized", help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type", type=str, default="uniform", help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta", type=float, default=0.0, help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--dyn", action="store_true", help="whether to activate the dynamic train/inference"
    )
    parser.add_argument(
        "--sequence", action="store_true"
    )
    parser.add_argument(
        "--select_step", type=int, default=None
    )
    parser.add_argument(
        "--select_depth", type=int, default=None
    )

    parser.add_argument(
        "--cache", action="store_true"
    )
    parser.add_argument(
        "--cache_interval", type=int, default=None,
    )
    parser.add_argument(
        "--non_uniform", action="store_true"
    )
    parser.add_argument(
        "--pow", type=float, default=None,
    )
    parser.add_argument(
        "--center", type=int, default=None,
    )
    parser.add_argument(
        "--branch", type=int, default=None,
    )
    args = parser.parse_args()
    # parse config file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    new_config.select_step = args.select_step
    new_config.select_depth = args.select_depth

    torch.backends.cudnn.benchmark = True

    return args, new_config


def main():
    args, config = parse_args_and_config()
    
    if args.dyn:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    else:
        accelerator = Accelerator()
    args.accelerator = accelerator

    #log_root_dir = "{}_runtime_log".format(args.config[8:-4])
    log_root_dir = "runtime_log"
    dataset = args.config[8:-4]
    if args.cache:
        if args.non_uniform:
            sub_dir_name = "{}_{}_cache_{}_pow_{}_center_{}".format(dataset, args.exp, args.cache_interval, args.pow, args.center)
        else:
            sub_dir_name = "{}_{}_cache_{}".format(dataset, args.exp, args.cache_interval)
    else:
        sub_dir_name = "{}".format(args.exp)

    if accelerator.is_main_process:
        logger = Logger(
            root_dir=log_root_dir,
            sub_name=sub_dir_name,
            config=args.__dict__, 
            append=(args.sample == True)
        )
        args.logger = logger

        args.logger.log("Writing log file to {}".format(args.logger.sub_dir))
        args.logger.log("Exp instance PID = {}".format(os.getpid()))
    else:
        args.logger = EmptyLogger(
            root_dir=log_root_dir,
            sub_name=sub_dir_name,
        )

    args.image_folder = args.logger.setup_image_folder("{}".format(args.image_folder))

    args.seed += accelerator.process_index      
    # set random seed
    set_random_seed(args.seed)
    try:
        if args.cache:
            from ddpm.runners.deepcache import Diffusion
            runner = Diffusion(args, config)
            runner.sample()
        else:
            from ddpm.runners.diffusion import Diffusion
            runner = Diffusion(args, config)
            runner.sample()
    except Exception:
        logging.error(traceback.format_exc())

    return 0


if __name__ == "__main__":
    main()
