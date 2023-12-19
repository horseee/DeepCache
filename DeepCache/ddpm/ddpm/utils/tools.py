import torch
import random
import numpy as np


def unwrap_module(state_dict):
    unwrap_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            unwrap_state_dict[key[7:]] = value
        else:
            unwrap_state_dict[key] = value
    return unwrap_state_dict

def save_and_set_random_status(seed=1234):
    rng_status = torch.get_rng_state()
    set_random_seed(seed)
    return rng_status

def restore_random_status(rng_status):
    torch.set_rng_state(rng_status)

def set_random_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  