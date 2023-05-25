import os
import argparse
from pathlib import Path

from diffusion.test_ddpm import JointMaskImageStableDiffusionTester

from config_pixel import image_size, bits, gray
from config_pixel import mask_unet, img_unet, mask_bit_diffusion_model, img_latent_diffusion_model
from config_pixel import vae, dataset

LOSS_CONFS = ["mse", "dice"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--milestone', type=int, required=True)
    parser.add_argument('--load-dir', type=str, required=True)
    parser.add_argument('--save-dir', type=str, required=True)
    parser.add_argument('--loss-conf', type=str, default="dice")
    parser.add_argument('--num-samples', type=int, default=2000)

    args = parser.parse_args()

    assert args.loss_conf in LOSS_CONFS, "loss-conf should choose in " + str(LOSS_CONFS)
    mask_bit_diffusion_model.loss_type = args.loss_conf

    tester = JointMaskImageStableDiffusionTester(
        mask_bit_diffusion_model = mask_bit_diffusion_model,
        img_latent_diffusion_model = img_latent_diffusion_model,
        joint_dataset = dataset,
        stable_vae = vae,
        results_folder = args.load_dir,
    )

    ckpt = args.milestone
    save_dir = args.save_dir + "_" + str(ckpt)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    tester.load(ckpt)
    print("Evaluating ckpt milestone {} ...".format(ckpt))
    tester.dataset_gen_test_pixel(save_dir = save_dir, num_samples = args.num_samples, ori_size = False)

