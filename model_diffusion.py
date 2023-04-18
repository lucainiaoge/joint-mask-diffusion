# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
import math
from functools import partial

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from bit_gaussian_ddpm import decimal_to_bits
from model_blocks import Residual, Upsample, Downsample, PreNorm
from model_blocks import SinusoidalPosEmb, RandomOrLearnedSinusoidalPosEmb
from model_blocks import Block, ResnetBlock, LinearAttention, Attention
from utils import exists, default, identity, uniform, prob_mask_like

# model, unconditioned
class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
    
# model, classifier-free guidance with mask and label controller
class Unet_conditional(nn.Module):
    def __init__(
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        mask_channels = 1, # for mask label
        num_classes = 1, # for class label
        mask_cond_bits = None,
        self_condition = False,
        cond_drop_prob = 0.5,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        bit_scale = 1.
    ):
        super().__init__()

        # classifier-free guidance settings
        self.cond_drop_prob = cond_drop_prob # the prob of using null-label during samping
        
        # determine dimensions
        
        self.channels = channels
        self.mask_channels = mask_channels
        self.mask_cond_bits = mask_cond_bits
        if exists(mask_cond_bits):
            self.mask_channels = 1
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)
        input_channels = input_channels + mask_channels

        init_dim = default(init_dim, dim)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:])) #e.g., [(1,8),(2,4),(4,2),(8,1)]

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # init conv
        
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding = 3)
        
        # time embeddings
        
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb, #(B,) -> (B,fourier_dim)
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # mask embeddings
        
        mask_dim = init_dim
        self.bit_scale = bit_scale
        if exists(mask_cond_bits):
            self.mask_cond_conv = nn.Conv2d(mask_cond_bits, mask_dim, 7, padding = 3)
        else:
            self.mask_cond_conv = nn.Conv2d(mask_channels, mask_dim, 7, padding = 3)
        
        # class embeddings
        
        self.classes_emb = nn.Embedding(num_classes, dim)
        self.null_classes_emb = nn.Parameter(torch.randn(dim))
        classes_dim = dim # dim * 4
        self.classes_mlp = nn.Sequential(
            nn.Linear(dim, classes_dim),
            nn.GELU(),
            nn.Linear(classes_dim, classes_dim)
        )
        
        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, class_emb_dim = classes_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, class_emb_dim = classes_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, class_emb_dim = classes_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, class_emb_dim = classes_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, class_emb_dim = classes_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim, class_emb_dim = classes_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv2d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim, mask_emb_dim = mask_dim, class_emb_dim = classes_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond = None, mask_cond = None, class_cond = None, cond_drop_prob = None):
        # x: (B,C,H,W), time: (B,), x_self_cond: (B,C,H,W), mask_cond: (B,C_m,H,W), class_cond: (B,)
        batch, H, W, device = x.shape[0], x.shape[2], x.shape[3], x.device
        
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1) # (B,2C,H,W)
            
        # mask_cond embeddings
        
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        mask_cond = default(
            mask_cond, 
            lambda: repeat(torch.zeros_like(x).sum(dim=1, keepdim=True), 'b 1 h w -> b c h w', c=self.mask_channels) - 1
        ) # (B,c_cond,H,W), all 1 for default condition
        
        _, C_mask, H_mask, W_mask = mask_cond.shape
        H_scale = H_mask / H
        W_scale = W_mask / W
        assert H_scale == W_scale, "mask and input image should have the same H-W proportion"
        if H_mask != H:
            mask_cond = F.interpolate(mask_cond, size=(H,W), mode='nearest')
        
        if exists(cond_drop_prob):
            if cond_drop_prob >= 1.0:
                mask_cond = mask_cond * 0.0 - 1.0 # null label: all -1
            elif cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device) # (B,)
                keep_mask = repeat(keep_mask, 'b -> b 1 1 1')
                default_mask = mask_cond * 0.0 - 1.0
                mask_cond = torch.where(keep_mask, mask_cond, default_mask)
                # torch.where(): if keep_mask[i] == True, return[i] = mask_cond[i], else return[i] = 1.0
        
        if exists(self.mask_cond_bits):
            m = decimal_to_bits(mask_cond, self.mask_cond_bits) * self.bit_scale # (B,1,H,W) -> (B,BITS,H,W)
        else:
            m = mask_cond
        
        m = self.mask_cond_conv(m) # (B,mask_dim,H,W)
        x = torch.cat((mask_cond, x), dim = 1) # (B,2C+1,H,W)
        
        # class embeddings
        if exists(class_cond):
            classes_emb = self.classes_emb(class_cond)
            if cond_drop_prob > 0:
                keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
                keep_mask = rearrange(keep_mask, 'b -> b 1')
                null_classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
                classes_emb = torch.where(keep_mask, classes_emb, null_classes_emb)
        else:
            classes_emb = repeat(self.null_classes_emb, 'd -> b d', b = batch)
            
        c = self.classes_mlp(classes_emb)
        
        # forward u-net
        x = self.init_conv(x) # (B,init_dim,H,W)
        r = x.clone()
        t = self.time_mlp(time) # (B,time_dim)
        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, None, c)
            h.append(x)

            x = block2(x, t, None, c)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, None, c)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, None, c)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, None, c)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t, None, c)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t, m, c)
        return self.final_conv(x)
    
    # guidance forward
    def forward_with_cond_scale(self, *args, cond_scale = 1., **kwargs):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale