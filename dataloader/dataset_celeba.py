# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py

from pathlib import Path
import os

from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset
from torchvision import transforms as T, utils
from torchvision.transforms import InterpolationMode
from PIL import Image

from tqdm.auto import tqdm

from utils import exists, convert_image_to_fn

# dataset classes

class UnlabelledImageDataset(Dataset):
    def __init__(
        self,
        folder,
        image_size,
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img) # shape for RGB img: (3, image_size, image_size)

class MaskImageDataset(Dataset):
# mask should be integers, with values chosen in {0,1,2,...,types_in_mask-1}
# in outputs, mask numbers are normalized between [0,1]: mask_out = mask_img * 256 / types_in_mask
    def __init__(
        self,
        mask_folder,
        image_size,
        image_folder = None,
        types_in_mask = 18, # 2 for binary mask; it cannot be greater than the largest value in mask
        exts = ['jpg', 'jpeg', 'png', 'tiff'],
        augment_horizontal_flip = False,
        convert_image_to = None
    ):
        super().__init__()
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.image_size = image_size
        self.types_in_mask = types_in_mask
        
        self.image_paths = []
        self.mask_paths = []
        for ext in exts:
            for mask_path in Path(f'{mask_folder}').glob(f'**/*.{ext}'):
                if type(self.image_folder) == str:
                    name = mask_path.stem
                    for ext2 in exts:
                        image_path = os.path.join(image_folder, name+'.'+ext2)
                        if os.path.exists(image_path):
                            self.mask_paths.append(mask_path)
                            self.image_paths.append(image_path)
                            break
                else:
                    self.mask_paths.append(mask_path)

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()
        self.image_transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])
        self.mask_transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size, interpolation=InterpolationMode.NEAREST),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.mask_paths)

    def __getitem__(self, index):
        mask = Image.open(self.mask_paths[index])
        mask = self.mask_transform(mask)
        mask = mask * 256 / self.types_in_mask # scale to [0,1]
        if type(self.image_folder) == str:
            img = Image.open(self.image_paths[index])
            img = self.image_transform(img)
            
            return mask, img # shape for 1-channel mask and RGB img: (1, image_size, image_size), (3, image_size, image_size)
        else:
            return mask
