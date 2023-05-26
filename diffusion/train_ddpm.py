# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
import sys
sys.path.append("..")

import os
from pathlib import Path
from multiprocessing import cpu_count

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as err:
    # Error handling
    pass

import math

import torch
from torch.optim import Adam
from ema_pytorch import EMA
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils

from accelerate import Accelerator

from tqdm.auto import tqdm
from utils import exists, cycle, has_int_squareroot, num_to_groups

debug = False

class JointMaskImageStableDiffusionTrainer(object):
    def __init__(
        self,
        mask_bit_diffusion_model,
        img_latent_diffusion_model,
        joint_dataset,
        stable_vae,
        *,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        use_lion = False,
        split_batches = True,
    ):
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no'
        )
        self.accelerator.native_amp = amp
        
        self.mask_bit_diffusion_model = mask_bit_diffusion_model
        self.img_latent_diffusion_model = img_latent_diffusion_model
        self.stable_vae = stable_vae
        device = self.accelerator.device
        self.stable_vae.to(device)

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        # optimizer

        optim_klass = Lion if use_lion else Adam
        self.opt_mask = optim_klass(mask_bit_diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        self.opt_img = optim_klass(img_latent_diffusion_model.parameters(), lr = train_lr, betas = adam_betas)
        
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema_mask = EMA(mask_bit_diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema_img = EMA(img_latent_diffusion_model, beta = ema_decay, update_every = ema_update_every)
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.mask_bit_diffusion_model, self.img_latent_diffusion_model, self.opt_mask, self.opt_img = self.accelerator.prepare(
            self.mask_bit_diffusion_model, self.img_latent_diffusion_model, self.opt_mask, self.opt_img
        )
        
        # dataset and dataloader
        self.ds = joint_dataset
        
        dl = DataLoader(self.ds, batch_size = self.batch_size, shuffle = True, pin_memory = True, num_workers = 0) #cpu_count())
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'mask_bit_diffusion_model': self.accelerator.get_state_dict(self.mask_bit_diffusion_model),
            'opt_mask': self.opt_mask.state_dict(),
            'ema_mask': self.ema_mask.state_dict(),
            'img_latent_diffusion_model': self.accelerator.get_state_dict(self.img_latent_diffusion_model),
            'opt_img': self.opt_img.state_dict(),
            'ema_img': self.ema_img.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': "test" #__version__
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        mask_bit_diffusion_model = self.accelerator.unwrap_model(self.mask_bit_diffusion_model)
        mask_bit_diffusion_model.load_state_dict(data['mask_bit_diffusion_model'])
        
        img_latent_diffusion_model = self.accelerator.unwrap_model(self.img_latent_diffusion_model)
        img_latent_diffusion_model.load_state_dict(data['img_latent_diffusion_model'])
        
        self.step = data['step']
        self.opt_mask.load_state_dict(data['opt_mask'])
        self.ema_mask.load_state_dict(data['ema_mask'])
        self.opt_img.load_state_dict(data['opt_img'])
        self.ema_img.load_state_dict(data['ema_img'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
    
    def sample(self, num_samples, save_dir, base_filename, mask_first = True):
        accelerator = self.accelerator
        device = accelerator.device
        
        mask_path = os.path.join(save_dir, base_filename + "-mask.png")
        img_path = os.path.join(save_dir, base_filename + "-img.png")
        
        self.ema_mask.ema_model.eval()
        self.ema_img.ema_model.eval()

        with torch.no_grad():
            batches = num_to_groups(self.num_samples, self.batch_size)
            if mask_first:
                all_masks_list = [
                    self.ema_mask.ema_model.sample(batch_size=n) for n in batches
                ]
                all_images_list = [
                    self.ema_img.ema_model.sample(
                        batch_size=mask_cond.shape[0], 
                        mask_cond=mask_cond, 
                        # class_cond=torch.ones(mask_cond.shape[0]).to(device).long()
                    ) for mask_cond in all_masks_list
                ]
                if debug and self.step % 100 == 0:
                    fig, ax = plt.subplots(2,2)
                    z_img_val = all_images_list[0]
                    ax[0,0].matshow(torch.permute(self.stable_vae.decode(z_img_val)[0].cpu(), (1, 2, 0)))
                    ax[0,1].matshow(torch.permute(z_img_val[0][0:1].cpu(), (1, 2, 0)))
                    ax[1,0].matshow(torch.permute(z_img_val[0][1:2].cpu(), (1, 2, 0)))
                    ax[1,1].matshow(torch.permute(z_img_val[0][2:3].cpu(), (1, 2, 0)))
            else:
                all_images_list = [
                    self.ema_img.ema_model.sample(
                        batch_size=n, 
                        # class_cond=torch.ones(n).to(device).long()
                    ) for n in batches
                ]
                if debug and self.step % 100 == 0:
                    fig, ax = plt.subplots(2,2)
                    z_img_val = all_images_list[0]
                    ax[0,0].matshow(torch.permute(self.stable_vae.decode(z_img_val)[0].cpu(), (1, 2, 0)))
                    ax[0,1].matshow(torch.permute(z_img_val[0][0:1].cpu(), (1, 2, 0)))
                    ax[1,0].matshow(torch.permute(z_img_val[0][1:2].cpu(), (1, 2, 0)))
                    ax[1,1].matshow(torch.permute(z_img_val[0][2:3].cpu(), (1, 2, 0)))
                all_masks_list = [
                    self.ema_mask.ema_model.sample(
                        batch_size=z_img.shape[0], 
                        mask_cond=z_img, 
                        # class_cond=torch.zeros(img.shape[0]).to(device).long()
                    ) for z_img in all_images_list
                ]
        
            all_masks_list = [mask.float() for mask in all_masks_list]
            all_images_list = [self.stable_vae.decode(z_img).float() for z_img in all_images_list]
        
        all_masks = torch.cat(all_masks_list, dim = 0)
        all_images = torch.cat(all_images_list, dim = 0)
        
        utils.save_image(all_masks, mask_path, nrow = int(math.sqrt(self.num_samples)))
        utils.save_image(all_images, img_path, nrow = int(math.sqrt(self.num_samples)))

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_mask_loss = 0.
                total_img_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    mask, img = next(self.dl)
                    mask = mask.to(device)
                    img = img.to(device)
                    with torch.no_grad():
                        # z_mask = self.stable_vae.encode_mask(mask).sample()
                        z_img = self.stable_vae.encode(img).sample()
                    
                        if debug and self.step % 1000 == 0:
                            fig, ax = plt.subplots(2,2)
                            ax[0,0].matshow(torch.permute(self.stable_vae.decode(z_img)[0].cpu(), (1, 2, 0)))
                            ax[0,1].matshow(torch.permute(z_img[0][0:1].cpu(), (1, 2, 0)))
                            ax[1,0].matshow(torch.permute(z_img[0][1:2].cpu(), (1, 2, 0)))
                            ax[1,1].matshow(torch.permute(z_img[0][2:3].cpu(), (1, 2, 0)))
                    
                    with self.accelerator.autocast():
                        mask_loss = self.mask_bit_diffusion_model(mask, mask_cond = z_img, class_cond = None)
                        mask_loss = mask_loss / self.gradient_accumulate_every
                        total_mask_loss += mask_loss.item()
                        
                        img_loss = self.img_latent_diffusion_model(z_img, mask_cond = mask, class_cond = None)
                        img_loss = img_loss / self.gradient_accumulate_every
                        total_img_loss += img_loss.item()
                        
                        loss = mask_loss + img_loss
                    
                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.mask_bit_diffusion_model.parameters(), 1.0)
                accelerator.clip_grad_norm_(self.img_latent_diffusion_model.parameters(), 1.0)
                pbar.set_description(f'mask_loss: {total_mask_loss:.4f}, img_loss: {total_img_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt_mask.step()
                self.opt_img.step()
                self.opt_mask.zero_grad()
                self.opt_img.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema_mask.to(device)
                    self.ema_img.to(device)
                    self.ema_mask.update()
                    self.ema_img.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.ema_mask.ema_model.eval()
                        self.ema_img.ema_model.eval()
                        mask_first_base_filename = f"mask-first-sample-{milestone}"
                        img_first_base_filename = f"img-first-sample-{milestone}"
                        self.sample(self.num_samples, self.results_folder, mask_first_base_filename, mask_first = True)
                        self.sample(self.num_samples, self.results_folder, img_first_base_filename, mask_first = False)
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        