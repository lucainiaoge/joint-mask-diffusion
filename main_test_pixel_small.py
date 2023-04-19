import os
from einops import rearrange

import torch
from pathlib import Path
from collections import namedtuple
from multiprocessing import cpu_count

from stable_diffusion_vae import StablePretrainedVAE, DownUpsampleVAE
from model_diffusion import Unet_conditional
from dataset_mri import ProstateTianfei
from dataset_celeba import MaskImageDataset
from gaussian_ddpm import CFGGaussianDiffusion
from bit_gaussian_ddpm import CFGBitDiffusion
from test_ddpm import JointMaskImageStableDiffusionTester

image_size = 384
bits = 8
gray = True

# vae = StablePretrainedVAE(gray = gray)
vae = DownUpsampleVAE(gray = gray, down_factor = 3)

mask_unet = Unet_conditional(
    dim = 8,
    dim_mults=(1, 2, 4, 8),
    channels = bits,
    mask_channels = vae.c, # for mask condition (as image)
    num_classes = 1, # for class label
    mask_cond_bits = None,
    self_condition = False,
    cond_drop_prob = 0.5
)

img_unet = Unet_conditional(
    dim = 64,
    dim_mults=(1, 2, 4, 8),
    channels = vae.c,
    mask_channels = 1, # for mask condition
    num_classes = 1, # for class label
    mask_cond_bits = bits,
    self_condition = False,
    cond_drop_prob = 0.5
)

mask_bit_diffusion_model = CFGBitDiffusion(
    mask_unet,
    image_size = image_size,
    bits = bits,
    timesteps = 20,   # number of sampling steps
)

img_latent_diffusion_model = CFGGaussianDiffusion(
    img_unet,
    image_size = image_size // vae.downsample_factor,
    timesteps = 100,   # number of sampling steps
)

dataset_dir = "ProstateMRI"
dataset = ProstateTianfei(base_path="ProstateMRI", image_size=image_size, train_ratio=0.8, split='test', transform=None, gray=gray)

save_dir = './gen_mri_img_pixel_gray'

tester = JointMaskImageStableDiffusionTester(
    mask_bit_diffusion_model = mask_bit_diffusion_model,
    img_latent_diffusion_model = img_latent_diffusion_model,
    joint_dataset = dataset,
    stable_vae = vae,
    results_folder = "./results_mri_pixel_gray",
)

tester.load(300)
tester.img_gen_test_pixel(save_dir = save_dir, num_samples = 1000, ori_size = False)
