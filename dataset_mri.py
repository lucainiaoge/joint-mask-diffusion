# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

from pathlib import Path
import os
import random
import cv2

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision import transforms as T, utils
from torchvision.transforms import InterpolationMode

from PIL import Image

import SimpleITK as sitk
# import nibabel as nib
# from nibabel.testing import data_path

from tqdm.auto import tqdm

from utils import exists, convert_image_to_fn

# dataset classes

subdir_list = [
    "BIDMC", "BMC", "HK", "I2CVB", "RUNMC", "UCL"
]

def convert_from_nii_to_png(img):
    high = np.quantile(img, 0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg = (newimg * 255).astype(np.uint8)
    return newimg

# https://github.com/tfzhou/FedFA/blob/b76c6e638d2a4046d850e574714c96c28e0140eb/pyfed/dataset/dataset.py#L77
class ProstateTianfei(Dataset):
    def __init__(self, base_path, image_size=384, train_ratio=0.6, split='train', transform=None, gray=False):
        channels = {'BIDMC': 3, 'HK': 3, 'I2CVB': 3, 'BMC': 3, 'RUNMC': 3, 'UCL': 3}
        self.channels = channels
        self.split = split
        self.train_ratio = train_ratio
        self.image_size = image_size
        self.gray = gray
        if not exists(image_size):
            self.image_size = 384

        images, labels = [], []
        filenames = []
        for site in channels.keys():
            sitedir = os.path.join(base_path, site)
            sample_list = sorted(os.listdir(sitedir))
            sample_list = [x for x in sample_list if 'segmentation.nii.gz' in x.lower()]
            for sample in sample_list:
                sampledir = os.path.join(sitedir, sample)
                if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
                    imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
                    label_v = sitk.ReadImage(sampledir)
                    image_v = sitk.ReadImage(imgdir)
                    label_v = sitk.GetArrayFromImage(label_v)
                    label_v[label_v > 1] = 1
                    image_v = sitk.GetArrayFromImage(image_v)
                    image_v = convert_from_nii_to_png(image_v)

                    for i in range(1, label_v.shape[0] - 1):
                        label = np.array(label_v[i, :, :])
                        if np.all(label == 0):
                            continue
                        image = np.array(image_v[i - 1:i + 2, :, :])
                        image = np.transpose(image, (1, 2, 0))

                        labels.append(label)
                        images.append(image)
                        filenames.append(site+"-"+sample[:6]+"-"+str(i))
                    
        labels = np.array(labels).astype(int)
        images = np.array(images)

        index_path = os.path.join(base_path, "{}-index.npy".format(site))
        if not os.path.exists(index_path):
            index = np.random.permutation(len(images)).tolist()
            np.save(index_path, index)
        else:
            index = np.load(index_path).tolist()

        labels = labels[index]
        images = images[index]

        trainlen = int(max(self.train_ratio * len(labels), 32))
        vallen = int(0.2 * len(labels))
        testlen = int(0.2 * len(labels))

        if split == 'train':
            self.images, self.labels, self.filenames = images[:trainlen], labels[:trainlen], filenames[:trainlen]
        elif split == 'valid':
            self.images, self.labels, self.filenames = images[trainlen:trainlen + vallen], labels[trainlen:trainlen + vallen], filenames[trainlen:trainlen + vallen]
        else:
            self.images, self.labels, self.filenames = images[-testlen:], labels[-testlen:], filenames[-testlen:]

        self.transform = transform
        self.labels = self.labels.astype(np.long).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if exists(self.image_size):
            image = cv2.resize(image, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, dsize=(self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image, (2, 0, 1))
            image = torch.Tensor(image)

            label = self.transform(label)
        
        image = torch.permute(torch.as_tensor(image).float(), (2, 0, 1)) / 256
        label = torch.as_tensor(label).float().unsqueeze(0)
        if self.gray:
            image = image.mean(dim=0, keepdim=True)
        return label, image
    
    def get_filename(self, idx):
        return self.filenames[idx]

    
def save_mri_to_images(mri_dataset, label_dir, img_dir):
    assert label_dir != img_dir, "image and label directory should be different"
    for idx in range(len(mri_dataset)):
        label, image = mri_dataset[idx]
        filename = mri_dataset.get_filename(idx)
        label_pth = os.path.join(label_dir, filename+".png")
        img_pth = os.path.join(img_dir, filename+".png")
        save_image(label, label_pth)
        save_image(image, img_pth)
        print("saved file "+filename)
    print("all done!")