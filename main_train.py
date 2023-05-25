import os
import argparse
from pathlib import Path

from diffusion.test_ddpm import JointMaskImageStableDiffusionTester

from config_pixel import image_size, bits, gray
from config_pixel import mask_unet, img_unet, mask_bit_diffusion_model, img_latent_diffusion_model
from config_pixel import vae, dataset

# from config_latent import image_size, bits, gray
# from config_latent import mask_unet, img_unet, mask_bit_diffusion_model, img_latent_diffusion_model
# from config_latent import vae, dataset

LOSS_CONFS = ["mse", "dice"]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, required=True)
    parser.add_argument('--milestone', type=int, default=0)
    parser.add_argument('--loss-conf', type=str, default="dice")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--train-steps', type=int, default=500000)
    parser.add_argument('--save-interval', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()

    assert args.loss_conf in LOSS_CONFS, "loss-conf should choose in " + str(LOSS_CONFS)
    mask_bit_diffusion_model.loss_type = args.loss_conf

    trainer = JointMaskImageStableDiffusionTrainer(
        mask_bit_diffusion_model = mask_bit_diffusion_model,
        img_latent_diffusion_model = img_latent_diffusion_model,
        joint_dataset = dataset,
        stable_vae = vae,
        train_batch_size = args.batch_size,
        gradient_accumulate_every = 1,
        train_lr = args.lr,
        train_num_steps = args.train_steps,
        ema_update_every = 50, #10
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = args.save_interval,
        num_samples = 16,#25,
        results_folder = args.ckpt_dir,
        amp = False,
        fp16 = False,
        use_lion = False,
        split_batches = True,
    )

    if args.milestone > 0:
        trainer.load(args.milestone) # for checkpoint continuing

    trainer.train()