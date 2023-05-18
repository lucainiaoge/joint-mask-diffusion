import os
from pathlib import Path

from diffusion.test_ddpm import JointMaskImageStableDiffusionTester

from config_pixel_dice import image_size, bits, gray
from config_pixel_dice import mask_unet, img_unet, mask_bit_diffusion_model, img_latent_diffusion_model
from config_pixel_dice import vae, dataset

ckpts = [50]

tester = JointMaskImageStableDiffusionTester(
    mask_bit_diffusion_model = mask_bit_diffusion_model,
    img_latent_diffusion_model = img_latent_diffusion_model,
    joint_dataset = dataset,
    stable_vae = vae,
    results_folder = "./results_mri_pixel_gray_dice",
)


for ckpt in ckpts:
    save_dir = './gen_dataset_mri_img_pixel_gray_dice_{}'.format(ckpt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tester.load(ckpt)
    print("Evaluating ckpt milestone {} ...".format(ckpt))
    # tester.img_gen_test_pixel(save_dir = save_dir, num_samples = 2000, ori_size = False)
    tester.dataset_gen_test_pixel(save_dir = save_dir, num_samples = 2000, ori_size = False)
