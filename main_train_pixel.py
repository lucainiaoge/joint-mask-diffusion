import os
from pathlib import Path

from diffusion.train_ddpm import JointMaskImageStableDiffusionTrainer

from config_pixel import image_size, bits, gray
from config_pixel import mask_unet, img_unet, mask_bit_diffusion_model, img_latent_diffusion_model
from config_pixel import vae, dataset


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
    results_folder = './results_mri_pixel_gray',
    amp = False,
    fp16 = False,
    use_lion = False,
    split_batches = True,
)

# trainer.load(21) # for checkpoint continuing

trainer.train()
