# https://github.com/lucidrains/bit-diffusion/blob/main/bit_diffusion/bit_diffusion.py

from collections import namedtuple
from functools import partial
import math
from random import random

import torch
from torch import nn
import torch.nn.functional as F
from torch.special import expm1
from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm

from utils import exists, default, identity, normalize_to_neg_one_to_one, unnormalize_to_zero_to_one

# bits no greater than 8

# convert to bit representations and back

def binary_vec_to_float(x):
    return (x - 0.5) * 2

def float_to_binary_vec(x):
    return x > 0

def decimal_to_bits(x, bits: int):
    """ expects image tensor ranging from 0 to 1, outputs bit tensor ranging from -1 to 1 """
    if len(x.shape) == 4 and bits > 1:
        max_num = 2**bits - 1
        device = x.device
        x = (x * max_num).int().clamp(0, max_num)
        mask = 2**torch.arange(bits - 1, -1, -1, device = device) * 2**(8-bits)

        mask = rearrange(mask, 'd -> d 1 1')
        x = rearrange(x, 'b c h w -> b c 1 h w')

        bits = ((x & mask) != 0).float()
        bits = rearrange(bits, 'b c d h w -> b (c d) h w')
        bits = bits * 2 - 1
        return bits
    
    elif bits == 1:
        return binary_vec_to_float(x)
    
    else:
        assert 0, "input tensor not suitable for decimal-bit conversion"

def bits_to_decimal(x, bits: int):
    """ expects bits from -1 to 1, outputs image tensor from 0 to 1 """
    if len(x.shape) == 4 and bits > 1:
        max_num = 2**bits - 1
        device = x.device
        x = (x > 0).int()
        mask = 2**(torch.arange(bits - 1, -1, -1, device = device, dtype = torch.int32)) * 2**(8-bits)

        mask = rearrange(mask, 'd -> d 1 1')
        x = rearrange(x, 'b (c d) h w -> b c d h w', d = bits)
        dec = reduce(x * mask, 'b c d h w -> b c h w', 'sum')
        return (dec / max_num).clamp(0., 1.)
    
    elif bits == 1:
        return float_to_binary_vec(x)
    
    else:
        assert 0, "input tensor not suitable for decimal-bit conversion"

# bit diffusion class

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def beta_linear_log_snr(t):
    return -torch.log(expm1(1e-4 + 10 * (t ** 2)))

def alpha_cosine_log_snr(t, s: float = 0.008):
    return -log((torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** -2) - 1, eps = 1e-5) # not sure if this accounts for beta being clipped to 0.999 in discrete version

def log_snr_to_alpha_sigma(log_snr):
    return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

class CFGBitDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        bits,
        timesteps = 1000,
        use_ddim = False,
        noise_schedule = 'cosine',
        time_difference = 0.,
        bit_scale = 1.,
        loss_type = "mse"
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.bits = bits

        self.image_size = image_size

        if noise_schedule == "linear":
            self.log_snr = beta_linear_log_snr
        elif noise_schedule == "cosine":
            self.log_snr = alpha_cosine_log_snr
        else:
            raise ValueError(f'invalid noise schedule {noise_schedule}')

        self.bit_scale = bit_scale

        self.timesteps = timesteps
        self.use_ddim = use_ddim

        # proposed in the paper, summed to time_next
        # as a way to fix a deficiency in self-conditioning and lower FID when the number of sampling timesteps is < 400

        self.time_difference = time_difference

        self.loss_type = loss_type

    @property
    def device(self):
        return next(self.model.parameters()).device

    def get_sampling_timesteps(self, batch, *, device):
        times = torch.linspace(1., 0., self.timesteps + 1, device = device)
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, mask_cond = None, class_cond = None, cond_scale = 3., time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)
        time_pairs = self.get_sampling_timesteps(batch, device = device)
        img = torch.randn(shape, device=device)
        x_start = None

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.timesteps):
            # add the time delay
            time_next = (time_next - self.time_difference).clamp(min = 0.)
            
            noise_cond = self.log_snr(time)

            # get predicted x0
            x_start = self.model.forward_with_cond_scale(
                x = img, 
                time = noise_cond, 
                x_self_cond = x_start, 
                mask_cond = mask_cond,
                class_cond = class_cond,
                cond_scale = cond_scale
            )

            # clip x0
            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get log(snr)
            log_snr = self.log_snr(time)
            log_snr_next = self.log_snr(time_next)
            log_snr, log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            # get alpha sigma of time and next time
            alpha, sigma = log_snr_to_alpha_sigma(log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(log_snr_next)

            # derive posterior mean and variance
            c = -expm1(log_snr - log_snr_next)

            mean = alpha_next * (img * (1 - c) / alpha + c * x_start)
            variance = (sigma_next ** 2) * c
            log_variance = log(variance)

            # get noise
            noise = torch.where(
                rearrange(time_next > 0, 'b -> b 1 1 1'),
                torch.randn_like(img),
                torch.zeros_like(img)
            )

            img = mean + (0.5 * log_variance).exp() * noise

        return bits_to_decimal(img, self.bits)

    @torch.no_grad()
    def ddim_sample(self, shape, mask_cond = None, class_cond = None, cond_scale = 3., time_difference = None):
        batch, device = shape[0], self.device

        time_difference = default(time_difference, self.time_difference)
        time_pairs = self.get_sampling_timesteps(batch, device = device)
        img = torch.randn(shape, device = device)
        x_start = None

        for times, times_next in tqdm(time_pairs, desc = 'sampling loop time step'):
            # get times and noise levels
            log_snr = self.log_snr(times)
            log_snr_next = self.log_snr(times_next)

            padded_log_snr, padded_log_snr_next = map(partial(right_pad_dims_to, img), (log_snr, log_snr_next))

            alpha, sigma = log_snr_to_alpha_sigma(padded_log_snr)
            alpha_next, sigma_next = log_snr_to_alpha_sigma(padded_log_snr_next)

            # add the time delay
            times_next = (times_next - time_difference).clamp(min = 0.)

            # predict x0
            x_start = self.model.forward_with_cond_scale(
                x = img, 
                time = log_snr, 
                x_self_cond = x_start, 
                mask_cond = mask_cond,
                class_cond = class_cond,
                cond_scale = cond_scale
            )

            # clip x0
            x_start.clamp_(-self.bit_scale, self.bit_scale)

            # get predicted noise
            pred_noise = (img - alpha * x_start) / sigma.clamp(min = 1e-8)

            # calculate x next
            img = x_start * alpha_next + pred_noise * sigma_next

        return bits_to_decimal(img, self.bits)

    @torch.no_grad()
    def sample(self, batch_size = 16, mask_cond = None, class_cond = None, cond_scale = 3.):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.ddpm_sample if not self.use_ddim else self.ddim_sample
        return sample_fn(
            shape = (batch_size, channels, image_size, image_size), 
            mask_cond = mask_cond, 
            class_cond = class_cond,
            cond_scale = cond_scale
        ).float()

    def binary_dice_loss(self, pred_bit, gt_logit, scale = 2):
        assert self.bits == 1, "should be binary bit diffusion (bit==1)"
        pred_logit = F.sigmoid(pred_bit)

        # input and target shapes must match
        assert pred_logit.size() == gt_logit .size(), "'pred' and 'gt' must have the same shape"

        pred_logit = rearrange(pred_logit, "b 1 h w -> b (h w)")
        gt_logit = rearrange(gt_logit, "b 1 h w -> b (h w)")
        gt_logit = gt_logit.float()

        # compute per channel Dice Coefficient
        intersect = (2 * pred_logit * gt_logit).sum(-1) + 1
        denominator = (pred_logit + gt_logit).sum(-1) + 1

        return scale * (1 - intersect / denominator).mean()

    def forward(self, img, mask_cond = None, class_cond = None):
        batch, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'

        # sample random times
        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)

        # convert image to bit representation
        img_ori = img * 1.0
        img = decimal_to_bits(img, self.bits) * self.bit_scale

        # noise sample
        noise = torch.randn_like(img)
        noise_level = self.log_snr(times)
        padded_noise_level = right_pad_dims_to(img, noise_level)
        alpha, sigma =  log_snr_to_alpha_sigma(padded_noise_level)

        noised_img = alpha * img + sigma * noise

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        self_cond = None
        if random() < 0.5:
            with torch.no_grad():
                self_cond = self.model(noised_img, noise_level).detach_()

        # predict and take gradient step
        pred = self.model(noised_img, noise_level, self_cond, mask_cond, class_cond)

        if self.bits == 1 and self.loss_type == "dice":
            return self.binary_dice_loss(pred, img_ori)
        else:
            return F.mse_loss(pred, img)


