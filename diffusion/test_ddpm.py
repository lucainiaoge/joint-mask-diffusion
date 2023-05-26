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

class JointMaskImageStableDiffusionTester(object):
    def __init__(
        self,
        mask_bit_diffusion_model,
        img_latent_diffusion_model,
        joint_dataset,
        stable_vae,
        *,
        results_folder = './results',
        ema_update_every = 10,
        ema_decay = 0.995,
    ):
        self.accelerator = Accelerator(
            split_batches = True,
            mixed_precision = 'fp16'
        )
        self.accelerator.native_amp = False
        
        self.mask_bit_diffusion_model = mask_bit_diffusion_model
        self.img_latent_diffusion_model = img_latent_diffusion_model
        self.stable_vae = stable_vae
        device = self.accelerator.device
        self.stable_vae.to(device)

        self.batch_size = 1
        
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema_mask = EMA(mask_bit_diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema_img = EMA(img_latent_diffusion_model, beta = ema_decay, update_every = ema_update_every)
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.mask_bit_diffusion_model, self.img_latent_diffusion_model = self.accelerator.prepare(
            self.mask_bit_diffusion_model, self.img_latent_diffusion_model
        )
        
        # dataset and dataloader
        self.ds = joint_dataset
        
        dl = DataLoader(self.ds, batch_size = 1, shuffle = True, pin_memory = True, num_workers = 0)
        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)
        
    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        mask_bit_diffusion_model = self.accelerator.unwrap_model(self.mask_bit_diffusion_model)
        mask_bit_diffusion_model.load_state_dict(data['mask_bit_diffusion_model'])
        
        img_latent_diffusion_model = self.accelerator.unwrap_model(self.img_latent_diffusion_model)
        img_latent_diffusion_model.load_state_dict(data['img_latent_diffusion_model'])
        
        self.step = data['step']
        self.ema_mask.load_state_dict(data['ema_mask'])
        self.ema_img.load_state_dict(data['ema_img'])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])
        
    def segment_instance(self, img): # single-batch
        z_img = self.stable_vae.encode(img).sample()
        mask_pred = self.ema_mask.ema_model.sample(batch_size=1, mask_cond=z_img)
        return mask_pred
    
    @torch.no_grad()
    def compute_dice(self, gt_mask, est_mask):
        gt_mask = (gt_mask > 0)
        est_mask = (est_mask > 0)
        intersection = torch.logical_and(gt_mask, est_mask).to(int).sum().cpu().item()
        gt_area = gt_mask.to(int).sum().cpu().item()
        mask_area = est_mask.to(int).sum().cpu().item()
        return 2.0*intersection/(gt_area+mask_area)
    
    @torch.no_grad()
    def compute_iou(self, gt_mask, est_mask):
        gt_mask = (gt_mask > 0)
        est_mask = (est_mask > 0)
        intersection = torch.logical_and(gt_mask, est_mask).to(int).sum().cpu().item()
        union = torch.logical_or(gt_mask, est_mask).to(int).sum().cpu().item()
        return 1.0*intersection/union
    
    @torch.no_grad()
    def segment_test(self, save_dir):
        accelerator = self.accelerator
        device = accelerator.device
        self.ema_mask.to(device)
        self.ema_img.to(device)
        self.step = 0
        DICEs = []
        IoUs = []
        
        with tqdm(initial = 0, total = len(self.ds), disable = not accelerator.is_main_process) as pbar:
            while self.step < len(self.ds):
                gt_mask, img = next(self.dl)
                gt_mask = gt_mask.to(device)
                img = img.to(device)
                
                est_mask = self.segment_instance(img)
                
                dice = self.compute_dice(gt_mask, est_mask)
                iou = self.compute_iou(gt_mask, est_mask)
                DICEs.append(dice)
                IoUs.append(iou)
                
                self.step += 1
                pbar.set_description(f'ave_dice: {sum(DICEs)/self.step:.4f}, ave_iou: {sum(IoUs)/self.step:.4f}')
                
                est_mask_path = os.path.join(save_dir, str(self.step) + "-est-testing-mask.png")
                gt_mask_path = os.path.join(save_dir, str(self.step) + "-gt-testing-mask.png")
                gt_img_path = os.path.join(save_dir, str(self.step) + "-gt-testing-img.png")
                
                utils.save_image(est_mask, est_mask_path, nrow = 1)
                utils.save_image(gt_mask, gt_mask_path, nrow = 1)
                utils.save_image(img, gt_img_path, nrow = 1)
                
                pbar.update(1)
    
    @torch.no_grad()
    def img_gen_test_pixel(self, save_dir, num_samples, ori_size = False):
        accelerator = self.accelerator
        device = accelerator.device
        self.ema_img.to(device)
        self.step = 0
        with tqdm(initial = 0, total = num_samples, disable = not accelerator.is_main_process) as pbar:
            while self.step < num_samples:
                img = self.ema_img.ema_model.sample(batch_size=1)
                self.step += 1
                pbar.set_description(f'sample {self.step}')
                
                img_path = os.path.join(save_dir, str(self.step) + "-gen-img.png")
                
                if ori_size:
                    img = self.stable_vae.decode(img).float()
                
                utils.save_image(img, img_path, nrow = 1)
                
                pbar.update(1)

    @torch.no_grad()
    def dataset_gen_test_pixel(self, save_dir, num_samples, mask_first = True, ori_size = False):
        accelerator = self.accelerator
        device = accelerator.device

        img_dir = os.path.join(save_dir, "imgs")
        mask_dir = os.path.join(save_dir, "masks")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        if not os.path.exists(mask_dir):
            os.makedirs(mask_dir)

        self.ema_img.to(device)
        self.ema_mask.to(device)
        self.step = 0
        with tqdm(initial = 0, total = num_samples, disable = not accelerator.is_main_process) as pbar:
            while self.step < num_samples:
                if mask_first:
                    mask = self.ema_mask.ema_model.sample(batch_size=1)
                    img = self.ema_img.ema_model.sample(
                        batch_size=1, 
                        mask_cond=mask, 
                    )
                else:
                    img = self.ema_img.ema_model.sample(batch_size=1)
                    mask = self.ema_mask.ema_model.sample(
                        batch_size=1, 
                        mask_cond=img
                    )

                self.step += 1
                pbar.set_description(f'sample {self.step}')
                
                img_path = os.path.join(img_dir, str(self.step) + "-gen.png")
                mask_path = os.path.join(mask_dir, str(self.step) + "-gen.png")
                
                if ori_size:
                    img = self.stable_vae.decode(img).float()
                    mask = self.stable_vae.decode(mask).float()
                
                utils.save_image(img, img_path, nrow = 1)
                utils.save_image(mask, mask_path, nrow = 1)
                
                pbar.update(1)
