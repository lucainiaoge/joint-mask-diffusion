from diffusion.stable_diffusion_vae import StablePretrainedVAE, DownUpsampleVAE
from diffusion.model_diffusion import Unet_conditional
from diffusion.gaussian_ddpm import CFGGaussianDiffusion
from diffusion.bit_gaussian_ddpm import CFGBitDiffusion
from dataloader.dataset_mri import ProstateTianfei
from dataloader.dataset_celeba import MaskImageDataset

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
