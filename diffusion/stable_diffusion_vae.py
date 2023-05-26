import sys
sys.path.append("..")

import math
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from typing import List, Optional, Tuple, Union
from diffusers import AutoencoderKL

LATENT_NUM_CHANNEL = 4
VAE_DOWNSAMPLE_FACTOR = 8

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """This is a helper function that allows to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators one can seed each batched size individually. If CPU generators are passed the tensor
    will always be created on CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                logger.info(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape, generator=generator, device=self.parameters.device, dtype=self.parameters.dtype
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar, dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var, dim=dims)

    def mode(self):
        return self.mean


class StablePretrainedVAE:
    def __init__(self, bias = -50, scale = 100, gray = False):
        super().__init__()
        self.bias = bias
        self.scale = scale
        self.c = LATENT_NUM_CHANNEL
        self.downsample_factor = VAE_DOWNSAMPLE_FACTOR
        self.vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="vae")
        self.gray = gray
    
    def encode(self, x: torch.FloatTensor, return_dict: bool = True):
        if x.shape[1] == 3:
            z_distrib = self.vae.encode(x, return_dict)[0]
        else:
            z_distrib = self.vae.encode(torch.cat([x,x,x],dim=1), return_dict)[0]
        normalized_mean = (z_distrib.mean - self.bias) / self.scale
        normalized_logvar = z_distrib.logvar - math.log(self.scale)
        gaussian_param = torch.cat([normalized_mean, normalized_logvar], dim = 1)
        return DiagonalGaussianDistribution(gaussian_param, deterministic=False)
    
    def encode_mask(self, x: torch.FloatTensor, return_dict: bool = True):
        return self.encode(x, return_dict)
    
    def decode(self, z: torch.FloatTensor, return_dict: bool = True):
        z_unnormalized = z * self.scale + self.bias
        if not self.gray:
            return self.vae.decode(z_unnormalized, return_dict)[0]
        else:
            return self.vae.decode(z_unnormalized, return_dict)[0].mean(dim=1, keepdim=True)
    def decode_mask(self, z: torch.FloatTensor, return_dict: bool = True):
        z_unnormalized = z * self.scale + self.bias
        return self.vae.decode(z_unnormalized, return_dict)[0].mean(dim=1,keepdim=True)
    
    def decode_bin_mask(self, z: torch.FloatTensor, return_dict: bool = True):
        z_unnormalized = z * self.scale + self.bias
        return self.vae.decode(z_unnormalized, return_dict)[0].mean(dim=1,keepdim=True) > 0.5
    
    def to(self, device):
        self.vae.to(device)

class DownUpsampleVAE(nn.Module):
    def __init__(self, gray = False, down_factor = VAE_DOWNSAMPLE_FACTOR):
        super().__init__()
        self.gray = gray
        if not gray:
            self.c = 3
        else:
            self.c = 1
        self.downsample_factor = down_factor
        
    def encode(self, x):
        batch, H, W, device = x.shape[0], x.shape[2], x.shape[3], x.device
        H_down = int(H / self.downsample_factor)
        W_down = int(W / self.downsample_factor)
        mu_z = F.interpolate(x, size=(H_down,W_down), mode='nearest')
        logvar_z = mu_z * 0 - 30
        gaussian_param = torch.cat([mu_z, logvar_z], dim = 1)
        return DiagonalGaussianDistribution(gaussian_param, deterministic=True)
        
    def decode(self, z):
        batch, H_down, W_down, device = z.shape[0], z.shape[2], z.shape[3], z.device
        H = int(H_down * self.downsample_factor)
        W = int(W_down * self.downsample_factor)
        return F.interpolate(z, size=(H,W), mode='nearest')
    
    def forward(self, x):
        z = self.encode(x).sample()
        return self.decode(z)
    
    def encode_mask(self, x):
        if self.c == 3:
            return self.encode(torch.cat([x,x,x],dim=1))
        elif self.c == 1:
            return self.encode(x)
    def decode_mask(self, z):
        return self.decode(z).mean(dim=1,keepdim=True)
    def decode_bin_mask(self, z):
        return self.decode(z).mean(dim=1,keepdim=True) > 0.5
