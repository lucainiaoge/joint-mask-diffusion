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
from train_ddpm import JointMaskImageStableDiffusionTrainer

image_size = 384
bits = 1
gray = True

# vae = StablePretrainedVAE(gray = gray)
vae = DownUpsampleVAE(gray = gray, down_factor = 3)

mask_unet = Unet_conditional(
    dim = 16, # 8
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
    timesteps = 40,   # number of sampling steps
    loss_type = "dice",
)

img_latent_diffusion_model = CFGGaussianDiffusion(
    img_unet,
    image_size = image_size // vae.downsample_factor,
    timesteps = 100,   # number of sampling steps
)


dataset_dir = "ProstateMRI"
dataset = ProstateTianfei(base_path="ProstateMRI", image_size=image_size, train_ratio=0.8, split='train', transform=None, gray=gray)

# celeba_folder = "CelebA_subset"
# mask_folder = os.path.join(celeba_folder, "mask")
# image_folder = os.path.join(celeba_folder, "ori")
# dataset = MaskImageDataset(mask_folder = mask_folder, image_size = image_size, image_folder = image_folder)

trainer = JointMaskImageStableDiffusionTrainer(
    mask_bit_diffusion_model = mask_bit_diffusion_model,
    img_latent_diffusion_model = img_latent_diffusion_model,
    joint_dataset = dataset,
    stable_vae = vae,
    train_batch_size = 8, #4
    gradient_accumulate_every = 1,
    train_lr = 1e-4,
    train_num_steps = 500000,
    ema_update_every = 50, #10
    ema_decay = 0.995,
    adam_betas = (0.9, 0.99),
    save_and_sample_every = 10000,#1000,
    num_samples = 16,#25,
    results_folder = './results_mri_pixel_gray_dice',
    amp = False,
    fp16 = False,
    use_lion = False,
    split_batches = True,
)

# trainer.load(134) #optional

trainer.train()

