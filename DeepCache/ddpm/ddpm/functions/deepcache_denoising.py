import torch

from scipy.stats import shapiro
import numpy as np

def sample_gaussian_centered(n=1000, sample_size=100, std_dev=100, shift=0):
    samples = []
    
    while len(samples) < sample_size:
        # Sample from a Gaussian centered at n/2
        sample = int(np.random.normal(loc=n/2+shift, scale=std_dev))
        
        # Check if the sample is in bounds
        if 1 <= sample < n and sample not in samples:
            samples.append(sample)
    
    return samples

def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace((-center)**(1/pow), (total_numbers-center)**(1/pow), n_samples+1)
        #print(x_values)
        #print([x for x in np.unique(np.int32(x_values**pow))[:-1]])
        # Raise these values to the power of 1.5 to get a non-linear distribution
        indices = [0] + [x+center for x in np.unique(np.int32(x_values**pow))[1:-1]]
        if len(indices) == n_samples:
            break
        
        pow -=0.02
    return indices, pow

def sample_from_quad(total_numbers, n_samples, pow=1.2):
    # Generate linearly spaced values between 0 and a max value
    x_values = np.linspace(0, total_numbers**(1/pow), n_samples+1)

    # Raise these values to the power of 1.5 to get a non-linear distribution
    indices = np.unique(np.int32(x_values**pow))[:-1]
    return indices

def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
    return a


def generalized_steps(x, seq, model, b, timesteps, cache_interval=None, non_uniform=False, pow=None, center=None,  branch=None, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        x0_preds = []
        xs = [x]
        prv_f = None

        cur_i = 0
        if non_uniform:
            num_slow = timesteps // cache_interval
            if timesteps % cache_interval > 0:
                num_slow += 1
            interval_seq, final_pow = sample_from_quad_center(total_numbers=timesteps, n_samples=num_slow, center=center, pow=pow)
        else:
            interval_seq = list(range(0, timesteps, cache_interval))
            interval = cache_interval
        #print(non_uniform, interval_seq)
        

        slow_path_count = 0
        save_features = []
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(b, t.long())
            at_next = compute_alpha(b, next_t.long())
            xt = xs[-1].to('cuda')

            with torch.no_grad():
                if cur_i in interval_seq: #%
                #if cur_i % interval == 0:
                    #print(cur_i, interval_seq)
                    et, cur_f = model(xt, t, prv_f=None,branch=branch)
                    prv_f = cur_f
                    save_features.append(cur_f[0].detach().cpu())
                    slow_path_count+= 1
                else:
                    et, cur_f = model(xt, t, prv_f=prv_f,branch=branch)
                    #quick_path_count+= 1

            #print(i, torch.mean(et) / torch.mean(xt), torch.var(et)/torch.var(xt), torch.mean(et), torch.var(et))

            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
            x0_preds.append(x0_t.to('cpu'))
            c1 = (
                kwargs.get("eta", 0) * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            )
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to('cpu'))

            cur_i += 1

    return xs, x0_preds


def ddpm_steps(x, seq, model, b, **kwargs):
    with torch.no_grad():
        n = x.size(0)
        seq_next = [-1] + list(seq[:-1])
        xs = [x]
        x0_preds = []
        betas = b
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = compute_alpha(betas, t.long())
            atm1 = compute_alpha(betas, next_t.long())
            beta_t = 1 - at / atm1
            x = xs[-1].to('cuda')

            output = model(x, t.float())
            e = output

            x0_from_e = (1.0 / at).sqrt() * x - (1.0 / at - 1).sqrt() * e
            x0_from_e = torch.clamp(x0_from_e, -1, 1)
            x0_preds.append(x0_from_e.to('cpu'))
            mean_eps = (
                (atm1.sqrt() * beta_t) * x0_from_e + ((1 - beta_t).sqrt() * (1 - atm1)) * x
            ) / (1.0 - at)

            mean = mean_eps
            noise = torch.randn_like(x)
            mask = 1 - (t == 0).float()
            mask = mask.view(-1, 1, 1, 1)
            logvar = beta_t.log()
            sample = mean + mask * torch.exp(0.5 * logvar) * noise
            xs.append(sample.to('cpu'))
    return xs, x0_preds
