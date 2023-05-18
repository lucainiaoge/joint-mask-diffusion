import os
from pathlib import Path

from diffusion.test_ddpm import JointMaskImageStableDiffusionTester

from config_pixel import image_size, bits, gray
from config_pixel import mask_unet, img_unet, mask_bit_diffusion_model, img_latent_diffusion_model
from config_pixel import vae, dataset

ckpts = [10, 20, 30, 40, 50]

tester = JointMaskImageStableDiffusionTester(
    mask_bit_diffusion_model = mask_bit_diffusion_model,
    img_latent_diffusion_model = img_latent_diffusion_model,
    joint_dataset = dataset,
    stable_vae = vae,
    results_folder = "./results_mri_pixel_gray",
)


for ckpt in ckpts:
    save_dir = './segmentation_mri_img_pixel_gray_{}'.format(ckpt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tester.load(ckpt)
    print("Evaluating (segmentation) ckpt milestone {} ...".format(ckpt))
    tester.segment_test(save_dir = save_dir)
